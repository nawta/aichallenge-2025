import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, ConcatDataset

logger = logging.getLogger(__name__)

# Normalization constants for kinematic state features
ODOM_NORM_CONSTANTS = {
    'position': 100.0,      # Normalize position by 100m (relative coordinates)
    'orientation': 1.0,     # Quaternion already in [-1, 1]
    'linear_vel': 30.0,     # Max speed ~30 m/s
    'angular_vel': 3.14159, # Max angular velocity ~π rad/s
}


def normalize_odom(odom: np.ndarray) -> np.ndarray:
    """
    Normalize kinematic state features.
    
    Input shape: (N, 13) or (13,)
    Features:
        0-2: position (x, y, z) -> /100.0
        3-6: orientation (qx, qy, qz, qw) -> unchanged
        7-9: linear_vel (vx, vy, vz) -> /30.0
        10-12: angular_vel (wx, wy, wz) -> /π
    
    Args:
        odom: Raw odometry features
        
    Returns:
        Normalized odometry features
    """
    odom = odom.copy().astype(np.float32)
    
    if odom.ndim == 1:
        # Single sample
        odom[0:3] /= ODOM_NORM_CONSTANTS['position']
        # odom[3:7] unchanged (quaternion)
        odom[7:10] /= ODOM_NORM_CONSTANTS['linear_vel']
        odom[10:13] /= ODOM_NORM_CONSTANTS['angular_vel']
    else:
        # Batch of samples (N, 13)
        odom[:, 0:3] /= ODOM_NORM_CONSTANTS['position']
        # odom[:, 3:7] unchanged (quaternion)
        odom[:, 7:10] /= ODOM_NORM_CONSTANTS['linear_vel']
        odom[:, 10:13] /= ODOM_NORM_CONSTANTS['angular_vel']
    
    return odom


class ScanControlSequenceDataset(Dataset):
    """
    A PyTorch Dataset for a single sequence of LiDAR scans and control commands.

    Loads synchronized .npy files (scans, steers, accelerations) from a specific
    directory. The LiDAR scans are normalized by the specified maximum range.
    
    Optionally loads odometry (kinematic state) data if use_odom=True.
    Supports mirror augmentation for data augmentation during training.
    Supports temporal sequence mode with seq_len > 1.

    Attributes:
        seq_dir (Path): Path to the sequence directory.
        max_range (float): Maximum range for LiDAR normalization.
        use_odom (bool): Whether to load and return odometry data.
        seq_len (int): Number of frames in each sequence (1 for single-frame mode).
        augment_mirror (bool): Whether to apply mirror augmentation.
        augment_prob (float): Probability of applying mirror augmentation.
        scans (np.ndarray): Normalized scan data array (N, num_points).
        steers (np.ndarray): Steering angle array (N,).
        accels (np.ndarray): Acceleration array (N,).
        odom (np.ndarray): Normalized odometry data array (N, 13) if use_odom=True.
    """

    def __init__(
        self, 
        seq_dir: Union[str, Path], 
        max_range: float = 30.0,
        use_odom: bool = False,
        seq_len: int = 1,
        augment_mirror: bool = True,
        augment_prob: float = 0.5
    ):
        """
        Initializes the dataset from a sequence directory.

        Args:
            seq_dir: Path to the directory containing .npy files.
            max_range: Maximum range value to normalize LiDAR data (0.0 to 1.0).
            use_odom: Whether to load odometry data (default: False for backward compatibility).
            seq_len: Number of frames per sequence (default: 1 for single-frame mode).
                     When > 1, returns sequences of consecutive frames.
            augment_mirror: Whether to apply mirror augmentation (default: True).
            augment_prob: Probability of applying mirror augmentation (default: 0.5).

        Raises:
            ValueError: If data lengths do not match or files are missing.
        """
        self.seq_dir = Path(seq_dir)
        self.max_range = max_range
        self.use_odom = use_odom
        self.seq_len = seq_len
        self.augment_mirror = augment_mirror
        self.augment_prob = augment_prob

        try:
            # Load raw data
            self.scans = np.load(self.seq_dir / "scans.npy")         # Shape: (N, num_points)
            self.steers = np.load(self.seq_dir / "steers.npy")       # Shape: (N,)
            self.accels = np.load(self.seq_dir / "accelerations.npy") # Shape: (N,)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required .npy files in {self.seq_dir}: {e}")

        # Validate data consistency
        n_samples = len(self.scans)
        if not (len(self.steers) == n_samples and len(self.accels) == n_samples):
            raise ValueError(
                f"Data length mismatch in {self.seq_dir}: "
                f"Scans={len(self.scans)}, Steers={len(self.steers)}, Accels={len(self.accels)}"
            )

        # Preprocessing: Clip and Normalize scans
        # Values are clipped to [0, max_range] and then scaled to [0, 1]
        self.scans = np.clip(self.scans, 0.0, self.max_range) / self.max_range

        # Load odometry data if requested
        if self.use_odom:
            odom_path = self.seq_dir / "odom.npy"
            if odom_path.exists():
                self.odom = np.load(odom_path)  # Shape: (N, 13)
                if len(self.odom) != n_samples:
                    raise ValueError(
                        f"Odom length mismatch in {self.seq_dir}: "
                        f"Scans={n_samples}, Odom={len(self.odom)}"
                    )
                # Normalize odometry features
                self.odom = normalize_odom(self.odom)
            else:
                logger.warning(f"No odom.npy found in {self.seq_dir}, using zeros.")
                self.odom = np.zeros((n_samples, 13), dtype=np.float32)

    def __len__(self) -> int:
        # Reduce length to account for sequence window
        return len(self.scans) - self.seq_len + 1

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
        """
        Retrieves a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Single-frame mode (seq_len=1):
                If use_odom=False: (scan, target)
                If use_odom=True: (scan, odom, target)
            
            Sequence mode (seq_len>1):
                If use_odom=False: (scans, target)
                If use_odom=True: (scans, odoms, target)
            
            scan(s): Normalized LiDAR scan data (float32).
                     Shape: (scan_dim,) for single-frame, (seq_len, scan_dim) for sequence.
            odom(s): Normalized kinematic state features (float32).
                     Shape: (13,) for single-frame, (seq_len, 13) for sequence.
            target: Control command vector [acceleration, steering] (float32) at the last frame.
        """
        # Determine if using mirror augmentation for this sample
        apply_mirror = self.augment_mirror and np.random.random() < self.augment_prob
        
        if self.seq_len > 1:
            # Sequence mode: return seq_len consecutive frames
            end_idx = idx + self.seq_len
            scans = self.scans[idx:end_idx].astype(np.float32)  # (seq_len, scan_dim)
            
            # Target is at the last frame
            accel = np.float32(self.accels[end_idx - 1])
            steer = np.float32(self.steers[end_idx - 1])
            
            # Mirror Augmentation: flip all scans and negate steering
            if apply_mirror:
                scans = np.flip(scans, axis=1).copy()
                steer = -steer
            
            target = np.array([accel, steer], dtype=np.float32)
            
            if self.use_odom:
                odoms = self.odom[idx:end_idx].astype(np.float32)  # (seq_len, 13)
                return scans, odoms, target
            else:
                return scans, target
        else:
            # Single-frame mode (original behavior)
            scan = self.scans[idx].astype(np.float32)
            
            accel = np.float32(self.accels[idx])
            steer = np.float32(self.steers[idx])
            
            # Mirror Augmentation: flip scan and negate steering
            if apply_mirror:
                scan = np.flip(scan).copy()
                steer = -steer
            
            target = np.array([accel, steer], dtype=np.float32)
            
            if self.use_odom:
                odom = self.odom[idx].astype(np.float32)
                return scan, odom, target
            else:
                return scan, target


class MultiSeqConcatDataset(ConcatDataset):
    """
    A PyTorch ConcatDataset that aggregates multiple SequenceDatasets.

    Automatically discovers valid sequence directories within a root directory.
    Supports filtering sequences using inclusion and exclusion keywords.
    Supports mirror augmentation for data augmentation during training.
    Supports temporal sequence mode with seq_len > 1.
    """

    def __init__(
        self, 
        dataset_root: Union[str, Path], 
        max_range: float = 30.0, 
        use_odom: bool = False,
        seq_len: int = 1,
        augment_mirror: bool = True,
        augment_prob: float = 0.5,
        include: Optional[List[str]] = None, 
        exclude: Optional[List[str]] = None
    ):
        """
        Initializes the concatenated dataset.

        Args:
            dataset_root: Root directory containing sequence folders.
            max_range: Maximum range for LiDAR normalization.
            use_odom: Whether to load odometry data (default: False).
            seq_len: Number of frames per sequence (default: 1 for single-frame mode).
            augment_mirror: Whether to apply mirror augmentation (default: True).
            augment_prob: Probability of applying mirror augmentation (default: 0.5).
            include: List of substrings; if provided, only directories containing
                     at least one of these substrings will be loaded.
            exclude: List of substrings; directories containing any of these
                     substrings will be skipped.

        Raises:
            RuntimeError: If no valid sequences are found after filtering.
        """
        dataset_root = Path(dataset_root)
        
        # Discover all subdirectories
        all_seq_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        target_seq_dirs = []

        # Apply filters
        for p in all_seq_dirs:
            name = p.name
            
            # Check inclusion criteria (OR logic)
            if include and not any(inc in name for inc in include):
                continue
            
            # Check exclusion criteria (OR logic)
            if exclude and any(exc in name for exc in exclude):
                continue
            
            target_seq_dirs.append(p)

        # Instantiate datasets
        datasets = []
        for seq_dir in target_seq_dirs:
            # Quick check for file existence before initialization
            required_files = ["scans.npy", "steers.npy", "accelerations.npy"]
            if all((seq_dir / f).exists() for f in required_files):
                try:
                    ds = ScanControlSequenceDataset(
                        seq_dir, 
                        max_range=max_range,
                        use_odom=use_odom,
                        seq_len=seq_len,
                        augment_mirror=augment_mirror,
                        augment_prob=augment_prob
                    )
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to load sequence {seq_dir}: {e}")
            else:
                logger.warning(f"Skipping {seq_dir.name}: Missing .npy files.")

        if not datasets:
            raise RuntimeError(f"No valid sequences found in {dataset_root} with provided filters.")

        super().__init__(datasets)
        odom_status = "with odom" if use_odom else "without odom"
        seq_status = f"seq_len={seq_len}" if seq_len > 1 else "single-frame"
        aug_status = f"mirror_aug={augment_mirror}(p={augment_prob})" if augment_mirror else "no_aug"
        logger.info(f"Loaded {len(datasets)} sequences {odom_status} {seq_status} {aug_status} from {dataset_root}. Total samples: {len(self)}")
