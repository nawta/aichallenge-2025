from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from lib.model import (
    TinyLidarNet, TinyLidarNetSmall, TinyLidarNetDeep, TinyLidarNetFusion,
    TinyLidarNetStacked, TinyLidarNetBiLSTM, TinyLidarNetTCN, TinyLidarNetMap,
    TinyLidarNetLocalBEV, TinyLidarNetGlobalBEV, TinyLidarNetDualBEV
)
from lib.data import MultiSeqConcatDataset, BEVMultiSeqConcatDataset
from lib.loss import WeightedSmoothL1Loss

# Temporal model names
TEMPORAL_MODELS = ["TinyLidarNetStacked", "TinyLidarNetBiLSTM", "TinyLidarNetTCN"]

# BEV model names
BEV_MODELS = ["TinyLidarNetLocalBEV", "TinyLidarNetGlobalBEV", "TinyLidarNetDualBEV"]


def load_map_image(map_path: str, device: torch.device) -> torch.Tensor:
    """Load and preprocess map image for TinyLidarNetMap.
    
    Args:
        map_path: Path to the map image file.
        device: Device to load the tensor to.
    
    Returns:
        Map image tensor of shape (1, 3, 224, 224).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])
    
    img = Image.open(map_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return img_tensor.to(device)



def clean_numerical_tensor(x: torch.Tensor) -> torch.Tensor:
    """NaN, infを安全に除去"""
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


@hydra.main(config_path="./config", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check model type
    model_name = cfg.model.name
    is_temporal = model_name in TEMPORAL_MODELS
    use_fusion = model_name == "TinyLidarNetFusion"
    use_map = model_name == "TinyLidarNetMap"
    is_bev = model_name in BEV_MODELS

    # Temporal models and Fusion model require odometry data
    use_odom = is_temporal or use_fusion or cfg.model.get("use_odom", False)

    # Sequence length for temporal models
    seq_len = cfg.model.get("seq_len", 10) if is_temporal else 1

    # BEV mode for BEV models
    if is_bev:
        if model_name == "TinyLidarNetLocalBEV":
            bev_mode = "local"
        elif model_name == "TinyLidarNetGlobalBEV":
            bev_mode = "global"
        else:  # TinyLidarNetDualBEV
            bev_mode = "both"
    else:
        bev_mode = None

    if is_temporal:
        print(f"[INFO] Using temporal model: {model_name} with seq_len={seq_len}")
    elif use_fusion:
        print("[INFO] Using TinyLidarNetFusion - loading odometry data")
    elif use_map:
        print("[INFO] Using TinyLidarNetMap - loading map image")
    elif is_bev:
        print(f"[INFO] Using BEV model: {model_name} with bev_mode='{bev_mode}'")

    # Data augmentation settings (defaults: ON)
    augment_mirror = cfg.data.get("augment_mirror", True)
    augment_prob = cfg.data.get("augment_prob", 0.5)
    
    if augment_mirror:
        print(f"[INFO] Mirror augmentation enabled (prob={augment_prob})")
    else:
        print("[INFO] Mirror augmentation disabled")

    # === Dataset ===
    if is_bev:
        # BEV models: use BEVMultiSeqConcatDataset
        lane_csv_path = cfg.model.lane_csv_path
        local_grid_size = cfg.model.get("local_grid_size", 64)
        local_resolution = cfg.model.get("local_resolution", 1.0)
        global_grid_size = cfg.model.get("global_grid_size", 128)
        global_resolution = cfg.model.get("global_resolution", 1.5)

        print(f"[INFO] Loading lane CSV from: {lane_csv_path}")
        print(f"[INFO] Local BEV: {local_grid_size}x{local_grid_size} @ {local_resolution}m/px")
        print(f"[INFO] Global BEV: {global_grid_size}x{global_grid_size} @ {global_resolution}m/px")

        # Training data: apply augmentation based on config
        train_dataset = BEVMultiSeqConcatDataset(
            cfg.data.train_dir,
            lane_csv_path=lane_csv_path,
            bev_mode=bev_mode,
            local_grid_size=local_grid_size,
            local_resolution=local_resolution,
            global_grid_size=global_grid_size,
            global_resolution=global_resolution,
            augment_mirror=augment_mirror,
            augment_prob=augment_prob
        )
        # Validation data: no augmentation for fair evaluation
        val_dataset = BEVMultiSeqConcatDataset(
            cfg.data.val_dir,
            lane_csv_path=lane_csv_path,
            bev_mode=bev_mode,
            local_grid_size=local_grid_size,
            local_resolution=local_resolution,
            global_grid_size=global_grid_size,
            global_resolution=global_resolution,
            augment_mirror=False,
            augment_prob=0.0
        )
    else:
        # Non-BEV models: use standard dataset
        # Training data: apply augmentation based on config
        train_dataset = MultiSeqConcatDataset(
            cfg.data.train_dir,
            use_odom=use_odom,
            seq_len=seq_len,
            augment_mirror=augment_mirror,
            augment_prob=augment_prob
        )
        # Validation data: no augmentation for fair evaluation
        val_dataset = MultiSeqConcatDataset(
            cfg.data.val_dir,
            use_odom=use_odom,
            seq_len=seq_len,
            augment_mirror=False,
            augment_prob=0.0
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # === Model ===
    state_dim = cfg.model.get("state_dim", 13)
    hidden_size = cfg.model.get("hidden_size", 128)
    
    if model_name == "TinyLidarNetSmall":
        model = TinyLidarNetSmall(
            input_dim=cfg.model.input_dim,
            output_dim=cfg.model.output_dim
        ).to(device)
    elif model_name == "TinyLidarNetDeep":
        model = TinyLidarNetDeep(
            input_dim=cfg.model.input_dim,
            output_dim=cfg.model.output_dim
        ).to(device)
    elif model_name == "TinyLidarNetFusion":
        model = TinyLidarNetFusion(
            input_dim=cfg.model.input_dim,
            state_dim=state_dim,
            output_dim=cfg.model.output_dim
        ).to(device)
    elif model_name == "TinyLidarNetStacked":
        model = TinyLidarNetStacked(
            input_dim=cfg.model.input_dim,
            state_dim=state_dim,
            seq_len=seq_len,
            output_dim=cfg.model.output_dim
        ).to(device)
    elif model_name == "TinyLidarNetBiLSTM":
        model = TinyLidarNetBiLSTM(
            input_dim=cfg.model.input_dim,
            state_dim=state_dim,
            hidden_size=hidden_size,
            output_dim=cfg.model.output_dim
        ).to(device)
    elif model_name == "TinyLidarNetTCN":
        num_levels = cfg.model.get("tcn_levels", 3)
        kernel_size = cfg.model.get("tcn_kernel_size", 3)
        causal = cfg.model.get("tcn_causal", False)  # Non-causal for training (sees future)
        model = TinyLidarNetTCN(
            input_dim=cfg.model.input_dim,
            state_dim=state_dim,
            hidden_size=hidden_size,
            num_levels=num_levels,
            kernel_size=kernel_size,
            causal=causal,
            output_dim=cfg.model.output_dim
        ).to(device)
    elif model_name == "TinyLidarNetMap":
        map_feature_dim = cfg.model.get("map_feature_dim", 128)
        model = TinyLidarNetMap(
            input_dim=cfg.model.input_dim,
            map_feature_dim=map_feature_dim,
            output_dim=cfg.model.output_dim
        ).to(device)

        # Load and cache map image
        map_image_path = cfg.model.get("map_image_path", None)
        if map_image_path:
            map_image = load_map_image(map_image_path, device)
            model.set_map_image(map_image)
            print(f"[INFO] Map image loaded and cached from: {map_image_path}")
        else:
            print("[WARN] No map_image_path specified - model will require map_image in forward()")
    elif model_name == "TinyLidarNetLocalBEV":
        model = TinyLidarNetLocalBEV(
            input_dim=cfg.model.input_dim,
            local_grid_size=cfg.model.get("local_grid_size", 64),
            local_channels=2,
            state_dim=state_dim,
            output_dim=cfg.model.output_dim
        ).to(device)
    elif model_name == "TinyLidarNetGlobalBEV":
        model = TinyLidarNetGlobalBEV(
            input_dim=cfg.model.input_dim,
            global_grid_size=cfg.model.get("global_grid_size", 128),
            global_channels=3,
            state_dim=state_dim,
            output_dim=cfg.model.output_dim
        ).to(device)
    elif model_name == "TinyLidarNetDualBEV":
        model = TinyLidarNetDualBEV(
            input_dim=cfg.model.input_dim,
            local_grid_size=cfg.model.get("local_grid_size", 64),
            local_channels=2,
            global_grid_size=cfg.model.get("global_grid_size", 128),
            global_channels=3,
            state_dim=state_dim,
            output_dim=cfg.model.output_dim
        ).to(device)
    else:
        model = TinyLidarNet(
            input_dim=cfg.model.input_dim,
            output_dim=cfg.model.output_dim
        ).to(device)

    if cfg.train.pretrained_path:
        model.load_state_dict(torch.load(cfg.train.pretrained_path))
        print(f"[INFO] Loaded pretrained model from {cfg.train.pretrained_path}")

    # === Loss & Optimizer ===
    criterion = WeightedSmoothL1Loss(
        steer_weight=cfg.train.loss.steer_weight,
        accel_weight=cfg.train.loss.accel_weight
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    # === Logging & Save dirs ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(cfg.train.save_dir).expanduser().resolve()
    log_dir = Path(cfg.train.log_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(log_dir / timestamp) as writer:
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = cfg.train.get("early_stop_patience", 10)

        best_path = save_dir / "best_model.pth"
        last_path = save_dir / "last_model.pth"

        # === Training Loop ===
        for epoch in range(cfg.train.epochs):
            model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{cfg.train.epochs}"):
                if is_temporal:
                    # Temporal models: (scans, odoms, targets) with sequence data
                    scans, odoms, targets = batch
                    # scans: (batch, seq_len, scan_dim)
                    # odoms: (batch, seq_len, state_dim)
                    scans = scans.to(device)
                    odoms = odoms.to(device)
                    targets = targets.to(device)
                    
                    scans = clean_numerical_tensor(scans)
                    odoms = clean_numerical_tensor(odoms)
                    targets = clean_numerical_tensor(targets)
                    
                    if model_name == "TinyLidarNetBiLSTM":
                        # Use bidirectional during training
                        outputs = model(scans, odoms, use_bidirectional=True)
                    else:
                        outputs = model(scans, odoms)
                elif use_odom:
                    # Fusion model: (scans, odom, targets)
                    scans, odom, targets = batch
                    scans = scans.unsqueeze(1).to(device)
                    odom = odom.to(device)
                    targets = targets.to(device)
                    
                    scans = clean_numerical_tensor(scans)
                    odom = clean_numerical_tensor(odom)
                    targets = clean_numerical_tensor(targets)
                    
                    outputs = model(scans, odom)
                elif use_map:
                    # Map model: (scans, targets) - uses cached map features
                    scans, targets = batch
                    scans = scans.unsqueeze(1).to(device)
                    targets = targets.to(device)

                    scans = clean_numerical_tensor(scans)
                    targets = clean_numerical_tensor(targets)

                    outputs = model(scans)  # Uses cached map features
                elif is_bev:
                    # BEV models: unpack based on bev_mode
                    if bev_mode == 'local':
                        # (scan, local_bev, odom, target)
                        scans, local_bev, odom, targets = batch
                        scans = scans.unsqueeze(1).to(device)
                        local_bev = local_bev.to(device)
                        odom = odom.to(device)
                        targets = targets.to(device)

                        scans = clean_numerical_tensor(scans)
                        local_bev = clean_numerical_tensor(local_bev)
                        odom = clean_numerical_tensor(odom)
                        targets = clean_numerical_tensor(targets)

                        outputs = model(scans, local_bev, odom)
                    elif bev_mode == 'global':
                        # (scan, global_bev, odom, target)
                        scans, global_bev, odom, targets = batch
                        scans = scans.unsqueeze(1).to(device)
                        global_bev = global_bev.to(device)
                        odom = odom.to(device)
                        targets = targets.to(device)

                        scans = clean_numerical_tensor(scans)
                        global_bev = clean_numerical_tensor(global_bev)
                        odom = clean_numerical_tensor(odom)
                        targets = clean_numerical_tensor(targets)

                        outputs = model(scans, global_bev, odom)
                    else:  # 'both'
                        # (scan, local_bev, global_bev, odom, target)
                        scans, local_bev, global_bev, odom, targets = batch
                        scans = scans.unsqueeze(1).to(device)
                        local_bev = local_bev.to(device)
                        global_bev = global_bev.to(device)
                        odom = odom.to(device)
                        targets = targets.to(device)

                        scans = clean_numerical_tensor(scans)
                        local_bev = clean_numerical_tensor(local_bev)
                        global_bev = clean_numerical_tensor(global_bev)
                        odom = clean_numerical_tensor(odom)
                        targets = clean_numerical_tensor(targets)

                        outputs = model(scans, local_bev, global_bev, odom)
                else:
                    # Standard model: (scans, targets)
                    scans, targets = batch
                    scans = scans.unsqueeze(1).to(device)
                    targets = targets.to(device)

                    scans = clean_numerical_tensor(scans)
                    targets = clean_numerical_tensor(targets)

                    outputs = model(scans)
                
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = validate(
                model, val_loader, device, criterion,
                use_odom=use_odom, is_temporal=is_temporal, use_map=use_map,
                is_bev=is_bev, bev_mode=bev_mode, model_name=model_name
            )

            print(f"Epoch {epoch+1:03d}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}")
            writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
            writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_path)
                print(f"[SAVE] Best model updated: {best_path} (val_loss={best_val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            torch.save(model.state_dict(), last_path)
            if patience_counter >= max_patience:
                print(f"[EarlyStop] No improvement for {max_patience} epochs.")
                break
    
    print("Training finished.")


def validate(
    model, loader, device, criterion,
    use_odom: bool = False, is_temporal: bool = False, use_map: bool = False,
    is_bev: bool = False, bev_mode: str = None, model_name: str = ""
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]", leave=False):
            if is_temporal:
                # Temporal models: (scans, odoms, targets)
                scans, odoms, targets = batch
                scans = scans.to(device)
                odoms = odoms.to(device)
                targets = targets.to(device)

                scans = clean_numerical_tensor(scans)
                odoms = clean_numerical_tensor(odoms)
                targets = clean_numerical_tensor(targets)

                if model_name == "TinyLidarNetBiLSTM":
                    # Use bidirectional during validation too (same as training)
                    outputs = model(scans, odoms, use_bidirectional=True)
                else:
                    outputs = model(scans, odoms)
            elif use_odom:
                # Fusion model: (scans, odom, targets)
                scans, odom, targets = batch
                scans = scans.unsqueeze(1).to(device)
                odom = odom.to(device)
                targets = targets.to(device)

                scans = clean_numerical_tensor(scans)
                odom = clean_numerical_tensor(odom)
                targets = clean_numerical_tensor(targets)

                outputs = model(scans, odom)
            elif use_map:
                # Map model: (scans, targets) - uses cached map features
                scans, targets = batch
                scans = scans.unsqueeze(1).to(device)
                targets = targets.to(device)

                scans = clean_numerical_tensor(scans)
                targets = clean_numerical_tensor(targets)

                outputs = model(scans)  # Uses cached map features
            elif is_bev:
                # BEV models: unpack based on bev_mode
                if bev_mode == 'local':
                    scans, local_bev, odom, targets = batch
                    scans = scans.unsqueeze(1).to(device)
                    local_bev = local_bev.to(device)
                    odom = odom.to(device)
                    targets = targets.to(device)

                    scans = clean_numerical_tensor(scans)
                    local_bev = clean_numerical_tensor(local_bev)
                    odom = clean_numerical_tensor(odom)
                    targets = clean_numerical_tensor(targets)

                    outputs = model(scans, local_bev, odom)
                elif bev_mode == 'global':
                    scans, global_bev, odom, targets = batch
                    scans = scans.unsqueeze(1).to(device)
                    global_bev = global_bev.to(device)
                    odom = odom.to(device)
                    targets = targets.to(device)

                    scans = clean_numerical_tensor(scans)
                    global_bev = clean_numerical_tensor(global_bev)
                    odom = clean_numerical_tensor(odom)
                    targets = clean_numerical_tensor(targets)

                    outputs = model(scans, global_bev, odom)
                else:  # 'both'
                    scans, local_bev, global_bev, odom, targets = batch
                    scans = scans.unsqueeze(1).to(device)
                    local_bev = local_bev.to(device)
                    global_bev = global_bev.to(device)
                    odom = odom.to(device)
                    targets = targets.to(device)

                    scans = clean_numerical_tensor(scans)
                    local_bev = clean_numerical_tensor(local_bev)
                    global_bev = clean_numerical_tensor(global_bev)
                    odom = clean_numerical_tensor(odom)
                    targets = clean_numerical_tensor(targets)

                    outputs = model(scans, local_bev, global_bev, odom)
            else:
                # Standard model: (scans, targets)
                scans, targets = batch
                scans = scans.unsqueeze(1).to(device)
                targets = targets.to(device)

                scans = clean_numerical_tensor(scans)
                targets = clean_numerical_tensor(targets)

                outputs = model(scans)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    main()
