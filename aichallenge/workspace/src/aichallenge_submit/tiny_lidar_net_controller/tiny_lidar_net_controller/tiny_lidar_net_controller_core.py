import logging
import numpy as np
from typing import Optional, Tuple, Union

from model.tinylidarnet import (
    TinyLidarNetNp, TinyLidarNetSmallNp, TinyLidarNetDeepNp, TinyLidarNetFusionNp,
    TinyLidarNetStackedNp, TinyLidarNetBiLSTMNp, TinyLidarNetTCNNp,
    TinyLidarNetLocalBEVNp, TinyLidarNetGlobalBEVNp, TinyLidarNetDualBEVNp
)


# Normalization constants for kinematic state features (must match training)
ODOM_NORM_CONSTANTS = {
    'position': 100.0,      # Normalize position by 100m (relative coordinates)
    'orientation': 1.0,     # Quaternion already in [-1, 1]
    'linear_vel': 30.0,     # Max speed ~30 m/s
    'angular_vel': 3.14159, # Max angular velocity ~π rad/s
}

# Temporal model architectures
TEMPORAL_ARCHITECTURES = ['stacked', 'bilstm', 'tcn']

# BEV-enabled model architectures
BEV_ARCHITECTURES = ['local_bev', 'global_bev', 'dual_bev']


class TinyLidarNetCore:
    """Core logic for the TinyLidarNet autonomous driving controller.

    This class manages the neural network model lifecycle, including initialization,
    weight loading, input preprocessing (cleaning, resizing, normalizing), and
    inference execution. It is designed to be framework-agnostic.

    Supports multiple architectures:
    - Single-frame: large, small, deep, fusion
    - Temporal: stacked, bilstm, tcn
    - BEV-enabled: local_bev, global_bev, dual_bev

    Attributes:
        input_dim (int): Dimension of the input vector expected by the model.
        output_dim (int): Dimension of the output vector (acceleration, steering).
        state_dim (int): Dimension of the kinematic state vector.
        seq_len (int): Sequence length for temporal models.
        hidden_size (int): Hidden size for temporal models (BiLSTM, TCN).
        architecture (str): Model architecture type.
        acceleration (float): Fixed acceleration value used in 'fixed' control mode.
        control_mode (str): Control strategy ('ai' or 'fixed').
        max_range (float): Maximum LiDAR range used for normalization and clipping.
        use_fusion (bool): Whether the model uses kinematic state fusion.
        is_temporal (bool): Whether the model is a temporal model.
        is_bev (bool): Whether the model uses BEV map inputs.
        model (object): The instantiated neural network model.
        logger (logging.Logger): Logger instance.
    """

    def __init__(
        self,
        input_dim: int = 1080,
        output_dim: int = 2,
        state_dim: int = 13,
        seq_len: int = 10,
        hidden_size: int = 128,
        architecture: str = 'large',
        ckpt_path: str = '',
        acceleration: float = 0.1,
        control_mode: str = 'ai',
        max_range: float = 30.0,
        local_bev_size: int = 64,
        local_bev_channels: int = 2,
        global_bev_size: int = 128,
        global_bev_channels: int = 3
    ):
        """Initializes the TinyLidarNetCore with specified parameters.

        Args:
            input_dim (int, optional): The number of LiDAR points expected by the model.
                Defaults to 1080.
            output_dim (int, optional): The number of output control values.
                Defaults to 2.
            state_dim (int, optional): The number of kinematic state features.
                Defaults to 13.
            seq_len (int, optional): Sequence length for temporal models.
                Defaults to 10.
            hidden_size (int, optional): Hidden size for temporal models.
                Defaults to 128.
            architecture (str, optional): The model architecture to use.
                Options: 'large', 'small', 'deep', 'fusion', 'stacked', 'bilstm', 'tcn',
                         'local_bev', 'global_bev', 'dual_bev'.
                Defaults to 'large'.
            ckpt_path (str, optional): Path to the numpy weight file (.npy or .npz).
                Defaults to ''.
            acceleration (float, optional): The constant acceleration value to apply
                when control_mode is set to 'fixed'. Defaults to 0.1.
            control_mode (str, optional): The control mode to determine output behavior.
                'ai' uses model output for both acceleration and steering.
                'fixed' uses the fixed acceleration value and model output for steering.
                Defaults to 'ai'.
            max_range (float, optional): The maximum range value for normalization.
                Values exceeding this will be clipped, and infinity will be replaced
                by this value. Defaults to 30.0.
            local_bev_size (int, optional): Size of local BEV grid. Defaults to 64.
            local_bev_channels (int, optional): Number of local BEV channels. Defaults to 2.
            global_bev_size (int, optional): Size of global BEV grid. Defaults to 128.
            global_bev_channels (int, optional): Number of global BEV channels. Defaults to 3.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.architecture = architecture.lower()
        self.acceleration = acceleration
        self.control_mode = control_mode.lower()
        self.max_range = max_range
        self.local_bev_size = local_bev_size
        self.local_bev_channels = local_bev_channels
        self.global_bev_size = global_bev_size
        self.global_bev_channels = global_bev_channels
        
        self.use_fusion = self.architecture == 'fusion'
        self.is_temporal = self.architecture in TEMPORAL_ARCHITECTURES
        self.is_bev = self.architecture in BEV_ARCHITECTURES
        self.logger = logging.getLogger(__name__)

        # Initialize model based on architecture
        if self.architecture == 'small':
            self.model = TinyLidarNetSmallNp(input_dim=self.input_dim, output_dim=self.output_dim)
        elif self.architecture == 'deep':
            self.model = TinyLidarNetDeepNp(input_dim=self.input_dim, output_dim=self.output_dim)
        elif self.architecture == 'fusion':
            self.model = TinyLidarNetFusionNp(
                input_dim=self.input_dim, 
                state_dim=self.state_dim,
                output_dim=self.output_dim
            )
        elif self.architecture == 'stacked':
            self.model = TinyLidarNetStackedNp(
                input_dim=self.input_dim,
                state_dim=self.state_dim,
                seq_len=self.seq_len,
                output_dim=self.output_dim
            )
        elif self.architecture == 'bilstm':
            self.model = TinyLidarNetBiLSTMNp(
                input_dim=self.input_dim,
                state_dim=self.state_dim,
                hidden_size=self.hidden_size,
                output_dim=self.output_dim
            )
        elif self.architecture == 'tcn':
            self.model = TinyLidarNetTCNNp(
                input_dim=self.input_dim,
                state_dim=self.state_dim,
                hidden_size=self.hidden_size,
                output_dim=self.output_dim
            )
        elif self.architecture == 'local_bev':
            self.model = TinyLidarNetLocalBEVNp(
                input_dim=self.input_dim,
                state_dim=self.state_dim,
                local_bev_size=self.local_bev_size,
                local_bev_channels=self.local_bev_channels,
                output_dim=self.output_dim
            )
        elif self.architecture == 'global_bev':
            self.model = TinyLidarNetGlobalBEVNp(
                input_dim=self.input_dim,
                state_dim=self.state_dim,
                global_bev_size=self.global_bev_size,
                global_bev_channels=self.global_bev_channels,
                output_dim=self.output_dim
            )
        elif self.architecture == 'dual_bev':
            self.model = TinyLidarNetDualBEVNp(
                input_dim=self.input_dim,
                state_dim=self.state_dim,
                local_bev_size=self.local_bev_size,
                local_bev_channels=self.local_bev_channels,
                global_bev_size=self.global_bev_size,
                global_bev_channels=self.global_bev_channels,
                output_dim=self.output_dim
            )
        else:
            self.model = TinyLidarNetNp(input_dim=self.input_dim, output_dim=self.output_dim)

        if ckpt_path:
            self._load_weights(ckpt_path)
        else:
            self.logger.warning("No weight file provided. Using randomly initialized weights.")

    def process(
        self, 
        ranges: np.ndarray, 
        odom: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Runs the complete inference pipeline on raw LiDAR data.

        This method handles data cleaning (NaN/Inf removal), resizing, normalization,
        and model inference. For fusion models, also processes kinematic state.

        Args:
            ranges (np.ndarray): A 1D numpy array containing raw LiDAR range data.
            odom (np.ndarray, optional): A 1D numpy array containing kinematic state
                features (13 dimensions). Required for fusion architecture.

        Returns:
            Tuple[float, float]: A tuple containing (acceleration, steering_angle).
                Values are clipped between -1.0 and 1.0.
        """
        # 1. Preprocess LiDAR (Clean -> Resize -> Normalize)
        processed_ranges = self._preprocess_ranges(ranges)

        # Prepare input tensor: (1, 1, input_dim)
        x = np.expand_dims(np.expand_dims(processed_ranges, axis=0), axis=1)

        # 2. Inference
        if self.use_fusion:
            # Process odometry data
            if odom is None:
                odom = np.zeros(self.state_dim, dtype=np.float32)
            
            processed_odom = self._preprocess_odom(odom)
            # Prepare state tensor: (1, state_dim)
            state = np.expand_dims(processed_odom, axis=0)
            
            outputs = self.model(x, state)[0]
        else:
            outputs = self.model(x)[0]

        # 3. Post-process
        if self.control_mode == "ai":
            accel = float(np.clip(outputs[0], -1.0, 1.0))
        else:
            accel = self.acceleration

        steer = float(np.clip(outputs[1], -1.0, 1.0))

        return accel, steer

    def process_sequence(
        self, 
        scan_seq: np.ndarray, 
        odom_seq: np.ndarray
    ) -> Tuple[float, float]:
        """Runs inference on a sequence of frames for temporal models.

        Args:
            scan_seq (np.ndarray): Sequence of LiDAR scans, shape (seq_len, scan_dim).
            odom_seq (np.ndarray): Sequence of odom data, shape (seq_len, state_dim).

        Returns:
            Tuple[float, float]: A tuple containing (acceleration, steering_angle).
        """
        if not self.is_temporal:
            raise ValueError("process_sequence() is only for temporal models")
        
        # Preprocess each frame in the sequence
        processed_scans = np.stack([
            self._preprocess_ranges(scan_seq[t]) for t in range(len(scan_seq))
        ], axis=0)  # (seq_len, input_dim)
        
        processed_odoms = np.stack([
            self._preprocess_odom(odom_seq[t]) for t in range(len(odom_seq))
        ], axis=0)  # (seq_len, state_dim)
        
        # Add batch dimension: (1, seq_len, dim)
        scans_batch = np.expand_dims(processed_scans, axis=0)
        odoms_batch = np.expand_dims(processed_odoms, axis=0)
        
        # Run inference based on architecture
        if self.architecture == 'bilstm':
            # BiLSTM processes frame by frame, maintaining state
            # For full sequence, we need to process all frames
            for t in range(len(scan_seq)):
                scan_t = scans_batch[:, t:t+1, :]  # (1, 1, scan_dim)
                odom_t = odoms_batch[:, t, :]      # (1, state_dim)
                outputs = self.model(scan_t, odom_t)
        else:
            # Stacked and TCN process the full sequence at once
            outputs = self.model(scans_batch, odoms_batch)
        
        outputs = outputs[0]  # Remove batch dimension
        
        # Post-process
        if self.control_mode == "ai":
            accel = float(np.clip(outputs[0], -1.0, 1.0))
        else:
            accel = self.acceleration

        steer = float(np.clip(outputs[1], -1.0, 1.0))

        return accel, steer

    def reset_temporal_state(self):
        """Resets the temporal state for BiLSTM model."""
        if self.architecture == 'bilstm' and hasattr(self.model, 'reset_state'):
            self.model.reset_state()

    def process_with_local_bev(
        self,
        ranges: np.ndarray,
        local_bev: np.ndarray,
        odom: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Runs inference with LiDAR, local BEV, and kinematic state (Pattern A).

        Args:
            ranges (np.ndarray): Raw LiDAR range data (1D array).
            local_bev (np.ndarray): Local BEV grid, shape (local_bev_channels, H, W).
            odom (np.ndarray, optional): Kinematic state features (13,).

        Returns:
            Tuple[float, float]: (acceleration, steering_angle).
        """
        if self.architecture != 'local_bev':
            raise ValueError("process_with_local_bev() requires 'local_bev' architecture")
        
        # Preprocess LiDAR
        processed_ranges = self._preprocess_ranges(ranges)
        x = np.expand_dims(np.expand_dims(processed_ranges, axis=0), axis=1)  # (1, 1, input_dim)
        
        # Prepare local BEV tensor: (1, channels, H, W)
        local_bev_batch = np.expand_dims(local_bev.astype(np.float32), axis=0)
        
        # Process odometry
        if odom is None:
            odom = np.zeros(self.state_dim, dtype=np.float32)
        processed_odom = self._preprocess_odom(odom)
        state = np.expand_dims(processed_odom, axis=0)  # (1, state_dim)
        
        # Inference
        outputs = self.model(x, local_bev_batch, state)[0]
        
        # Post-process
        accel = float(np.clip(outputs[0], -1.0, 1.0)) if self.control_mode == "ai" else self.acceleration
        steer = float(np.clip(outputs[1], -1.0, 1.0))
        
        return accel, steer

    def process_with_global_bev(
        self,
        ranges: np.ndarray,
        global_bev: np.ndarray,
        odom: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Runs inference with LiDAR, global BEV, and kinematic state (Pattern B).

        Args:
            ranges (np.ndarray): Raw LiDAR range data (1D array).
            global_bev (np.ndarray): Global BEV grid, shape (global_bev_channels, H, W).
            odom (np.ndarray, optional): Kinematic state features (13,).

        Returns:
            Tuple[float, float]: (acceleration, steering_angle).
        """
        if self.architecture != 'global_bev':
            raise ValueError("process_with_global_bev() requires 'global_bev' architecture")
        
        # Preprocess LiDAR
        processed_ranges = self._preprocess_ranges(ranges)
        x = np.expand_dims(np.expand_dims(processed_ranges, axis=0), axis=1)
        
        # Prepare global BEV tensor: (1, channels, H, W)
        global_bev_batch = np.expand_dims(global_bev.astype(np.float32), axis=0)
        
        # Process odometry
        if odom is None:
            odom = np.zeros(self.state_dim, dtype=np.float32)
        processed_odom = self._preprocess_odom(odom)
        state = np.expand_dims(processed_odom, axis=0)
        
        # Inference
        outputs = self.model(x, global_bev_batch, state)[0]
        
        # Post-process
        accel = float(np.clip(outputs[0], -1.0, 1.0)) if self.control_mode == "ai" else self.acceleration
        steer = float(np.clip(outputs[1], -1.0, 1.0))
        
        return accel, steer

    def process_with_dual_bev(
        self,
        ranges: np.ndarray,
        local_bev: np.ndarray,
        global_bev: np.ndarray,
        odom: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Runs inference with LiDAR, both BEVs, and kinematic state (Pattern C).

        Args:
            ranges (np.ndarray): Raw LiDAR range data (1D array).
            local_bev (np.ndarray): Local BEV grid, shape (local_bev_channels, H, W).
            global_bev (np.ndarray): Global BEV grid, shape (global_bev_channels, H, W).
            odom (np.ndarray, optional): Kinematic state features (13,).

        Returns:
            Tuple[float, float]: (acceleration, steering_angle).
        """
        if self.architecture != 'dual_bev':
            raise ValueError("process_with_dual_bev() requires 'dual_bev' architecture")
        
        # Preprocess LiDAR
        processed_ranges = self._preprocess_ranges(ranges)
        x = np.expand_dims(np.expand_dims(processed_ranges, axis=0), axis=1)
        
        # Prepare BEV tensors: (1, channels, H, W)
        local_bev_batch = np.expand_dims(local_bev.astype(np.float32), axis=0)
        global_bev_batch = np.expand_dims(global_bev.astype(np.float32), axis=0)
        
        # Process odometry
        if odom is None:
            odom = np.zeros(self.state_dim, dtype=np.float32)
        processed_odom = self._preprocess_odom(odom)
        state = np.expand_dims(processed_odom, axis=0)
        
        # Inference
        outputs = self.model(x, local_bev_batch, global_bev_batch, state)[0]
        
        # Post-process
        accel = float(np.clip(outputs[0], -1.0, 1.0)) if self.control_mode == "ai" else self.acceleration
        steer = float(np.clip(outputs[1], -1.0, 1.0))
        
        return accel, steer

    def process_with_bev(
        self,
        ranges: np.ndarray,
        local_bev: Optional[np.ndarray] = None,
        global_bev: Optional[np.ndarray] = None,
        odom: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Unified BEV processing method that routes to the appropriate processor.

        Args:
            ranges (np.ndarray): Raw LiDAR range data.
            local_bev (np.ndarray, optional): Local BEV grid.
            global_bev (np.ndarray, optional): Global BEV grid.
            odom (np.ndarray, optional): Kinematic state features.

        Returns:
            Tuple[float, float]: (acceleration, steering_angle).
        """
        if self.architecture == 'local_bev':
            if local_bev is None:
                raise ValueError("local_bev is required for 'local_bev' architecture")
            return self.process_with_local_bev(ranges, local_bev, odom)
        elif self.architecture == 'global_bev':
            if global_bev is None:
                raise ValueError("global_bev is required for 'global_bev' architecture")
            return self.process_with_global_bev(ranges, global_bev, odom)
        elif self.architecture == 'dual_bev':
            if local_bev is None or global_bev is None:
                raise ValueError("Both local_bev and global_bev are required for 'dual_bev' architecture")
            return self.process_with_dual_bev(ranges, local_bev, global_bev, odom)
        else:
            raise ValueError(f"process_with_bev() requires a BEV architecture, got '{self.architecture}'")

    def _load_weights(self, path: str) -> None:
        """Loads model weights from a file into the model parameters.

        Args:
            path (str): Path to the .npy or .npz weight file.

        Raises:
            ValueError: If the weight file format is unsupported.
            IOError: If the file cannot be read.
        """
        try:
            weights = np.load(path, allow_pickle=True)

            if isinstance(weights, np.lib.npyio.NpzFile):
                weight_dict = dict(weights.items())
            elif isinstance(weights, np.ndarray) and weights.dtype == object:
                weight_dict = weights.item()
            elif isinstance(weights, dict):
                weight_dict = weights
            else:
                raise ValueError(f"Unsupported weight format type: {type(weights)}")

            loaded_count = 0
            for key, value in weight_dict.items():
                key_norm = key.replace('.', '_')

                if key_norm in self.model.params:
                    self.model.params[key_norm] = value
                    loaded_count += 1

            self.logger.info(f"Successfully loaded {loaded_count} parameters from {path}")

        except Exception as e:
            self.logger.error(f"Failed to load weights from {path}: {e}")
            raise e

    def _preprocess_ranges(self, ranges: np.ndarray) -> np.ndarray:
        """Cleans, resizes, and normalizes LiDAR ranges.

        This method performs the following operations:
        1. Replaces NaNs with 0.0.
        2. Replaces infinite values with `self.max_range`.
        3. Clips all values to the range [0.0, `self.max_range`].
        4. Resizes the array to match `self.input_dim` via interpolation or padding.
        5. Normalizes the data by dividing by `self.max_range`.

        Args:
            ranges (np.ndarray): Source LiDAR range data.

        Returns:
            np.ndarray: Processed data array of shape (self.input_dim,).
        """
        # Work on a copy to avoid side effects on the input array
        ranges = ranges.copy()
        
        # Handle invalid values
        ranges[np.isnan(ranges)] = 0.0
        ranges[np.isinf(ranges)] = self.max_range
        
        # Clip to ensure data is within the expected range
        ranges = np.clip(ranges, 0.0, self.max_range)

        # Resize input if necessary
        current_len = len(ranges)
        if current_len > self.input_dim:
            idx = np.linspace(0, current_len - 1, self.input_dim, dtype=int)
            ranges = ranges[idx]
        elif current_len < self.input_dim:
            ranges = np.pad(ranges, (0, self.input_dim - current_len), 'constant')

        # Normalize
        return ranges / self.max_range

    def _preprocess_odom(self, odom: np.ndarray) -> np.ndarray:
        """Normalizes kinematic state features.

        This method normalizes the 13-dimensional kinematic state vector:
        - Position (0-2): Divided by 100.0
        - Orientation (3-6): Unchanged (quaternion already in [-1, 1])
        - Linear velocity (7-9): Divided by 30.0
        - Angular velocity (10-12): Divided by π

        Args:
            odom (np.ndarray): Raw kinematic state features (13,).

        Returns:
            np.ndarray: Normalized state features (13,).
        """
        odom = odom.copy().astype(np.float32)
        
        # Handle invalid values
        odom = np.nan_to_num(odom, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize each component
        odom[0:3] /= ODOM_NORM_CONSTANTS['position']
        # odom[3:7] unchanged (quaternion)
        odom[7:10] /= ODOM_NORM_CONSTANTS['linear_vel']
        odom[10:13] /= ODOM_NORM_CONSTANTS['angular_vel']
        
        return odom
