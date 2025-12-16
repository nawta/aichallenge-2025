#!/usr/bin/env python3
"""
TinyLidarNet - Train All Models Script
======================================

This script trains all model architectures with and without data augmentation.
Run this overnight to get all checkpoints ready in the morning.

Usage:
    python3 train_all_models.py                    # Use GPU
    python3 train_all_models.py --cpu              # Use CPU only
    python3 train_all_models.py --models TinyLidarNet TinyLidarNetDeep  # Specific models
    python3 train_all_models.py --skip-noaug       # Skip non-augmented training
    python3 train_all_models.py --epochs 50        # Custom epochs

Output:
    checkpoints/
    ├── TinyLidarNet_aug/
    ├── TinyLidarNet_noaug/
    └── ...
    weights/
    ├── TinyLidarNet_aug.npy
    └── ...
"""

import argparse
import subprocess
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()

# Model configurations
MODELS_CONFIG = {
    # Single-frame models
    "TinyLidarNet": {
        "convert_name": "tinylidarnet",
        "extra_args": {},
    },
    "TinyLidarNetSmall": {
        "convert_name": "tinylidarnet_small",
        "extra_args": {},
    },
    "TinyLidarNetDeep": {
        "convert_name": "tinylidarnet_deep",
        "extra_args": {},
    },
    "TinyLidarNetFusion": {
        "convert_name": "tinylidarnet_fusion",
        "extra_args": {"model.state_dim": 13},
    },
    # Temporal models
    "TinyLidarNetStacked": {
        "convert_name": "tinylidarnet_stacked",
        "extra_args": {"model.seq_len": 10},
        "convert_extra": ["--seq-len", "10"],
    },
    "TinyLidarNetBiLSTM": {
        "convert_name": "tinylidarnet_bilstm",
        "extra_args": {"model.seq_len": 10, "model.hidden_size": 128},
        "convert_extra": ["--seq-len", "10", "--hidden-size", "128"],
    },
    "TinyLidarNetTCN": {
        "convert_name": "tinylidarnet_tcn",
        "extra_args": {"model.seq_len": 10, "model.hidden_size": 128, "model.tcn_causal": False},
        "convert_extra": ["--seq-len", "10", "--hidden-size", "128"],
    },
}

# =============================================================================
# Helper Functions
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def run_command(cmd: List[str], log_file: Path, env: Optional[Dict] = None) -> bool:
    """Run a command and log output to file."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    with open(log_file, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=full_env,
            text=True,
            bufsize=1,
        )
        
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        
        process.wait()
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Exit code: {process.returncode}\n")
        f.write(f"Finished: {datetime.now()}\n")
    
    return process.returncode == 0


def train_model(
    model_name: str,
    config: Dict,
    augment: bool,
    checkpoint_base: Path,
    weights_dir: Path,
    log_dir: Path,
    use_cpu: bool,
    epochs: int,
    timestamp: str,
) -> Dict:
    """Train a single model configuration."""
    
    aug_suffix = "_aug" if augment else "_noaug"
    save_dir = checkpoint_base / f"{model_name}{aug_suffix}"
    log_file = log_dir / f"{model_name}{aug_suffix}_{timestamp}.log"
    weight_file = weights_dir / f"{model_name}{aug_suffix}.npy"
    
    result = {
        "model": model_name,
        "augment": augment,
        "save_dir": str(save_dir),
        "weight_file": str(weight_file),
        "status": "pending",
        "duration": 0,
    }
    
    print()
    print("━" * 50)
    print(f"🚀 Training: {model_name}{aug_suffix}")
    print(f"   Augmentation: {augment}")
    print(f"   Save Dir: {save_dir}")
    print(f"   Started: {datetime.now()}")
    print("━" * 50)
    
    # Build training command
    cmd = [
        "python3", str(SCRIPT_DIR / "train.py"),
        f"model.name={model_name}",
        f"data.augment_mirror={str(augment).lower()}",
        f"train.save_dir={save_dir}",
        f"train.epochs={epochs}",
    ]
    
    # Add model-specific arguments
    for key, value in config.get("extra_args", {}).items():
        if isinstance(value, bool):
            cmd.append(f"{key}={str(value).lower()}")
        else:
            cmd.append(f"{key}={value}")
    
    # Set environment
    env = {}
    if use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run training
    start_time = time.time()
    success = run_command(cmd, log_file, env)
    duration = time.time() - start_time
    result["duration"] = duration
    
    if success:
        print(f"✅ Training completed in {format_duration(duration)}")
        result["status"] = "trained"
        
        # Convert weights
        best_model_path = save_dir / "best_model.pth"
        if best_model_path.exists():
            print("📦 Converting weights...")
            
            convert_cmd = [
                "python3", str(SCRIPT_DIR / "convert_weight.py"),
                "--model", config["convert_name"],
                "--ckpt", str(best_model_path),
                "--output", str(weight_file),
            ]
            
            # Add convert-specific arguments
            convert_cmd.extend(config.get("convert_extra", []))
            
            if run_command(convert_cmd, log_dir / f"convert_{model_name}{aug_suffix}.log", env):
                print(f"✅ Weights saved: {weight_file}")
                result["status"] = "completed"
            else:
                print("❌ Weight conversion failed")
                result["status"] = "convert_failed"
        else:
            print("⚠️  No best_model.pth found")
            result["status"] = "no_checkpoint"
    else:
        print(f"❌ Training failed! Check log: {log_file}")
        result["status"] = "failed"
    
    print(f"   Finished: {datetime.now()}")
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train all TinyLidarNet models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU only")
    parser.add_argument("--models", nargs="+", choices=list(MODELS_CONFIG.keys()),
                        help="Specific models to train (default: all)")
    parser.add_argument("--skip-aug", action="store_true", help="Skip augmented training")
    parser.add_argument("--skip-noaug", action="store_true", help="Skip non-augmented training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR,
                        help="Output directory for checkpoints and weights")
    
    args = parser.parse_args()
    
    # Setup directories
    checkpoint_base = args.output_dir / "checkpoints"
    weights_dir = args.output_dir / "weights"
    log_dir = args.output_dir / "training_logs"
    
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine which models to train
    models_to_train = args.models or list(MODELS_CONFIG.keys())
    
    # Determine augmentation settings
    augment_settings = []
    if not args.skip_aug:
        augment_settings.append(True)
    if not args.skip_noaug:
        augment_settings.append(False)
    
    if not augment_settings:
        print("❌ Error: Cannot skip both augmented and non-augmented training")
        sys.exit(1)
    
    # Calculate total training runs
    total_runs = len(models_to_train) * len(augment_settings)
    
    print("=" * 60)
    print("TinyLidarNet Training Session")
    print(f"Started: {datetime.now()}")
    print("=" * 60)
    print()
    print(f"📋 Training Plan:")
    print(f"   Models: {len(models_to_train)}")
    print(f"   Augmentation variants: {len(augment_settings)}")
    print(f"   Total training runs: {total_runs}")
    print(f"   Epochs per run: {args.epochs}")
    print(f"   Device: {'CPU' if args.cpu else 'GPU'}")
    print()
    
    # Run training
    results = []
    completed = 0
    
    for model_name in models_to_train:
        config = MODELS_CONFIG[model_name]
        
        for augment in augment_settings:
            completed += 1
            print(f"\n[{completed}/{total_runs}] ", end="")
            
            result = train_model(
                model_name=model_name,
                config=config,
                augment=augment,
                checkpoint_base=checkpoint_base,
                weights_dir=weights_dir,
                log_dir=log_dir,
                use_cpu=args.cpu,
                epochs=args.epochs,
                timestamp=timestamp,
            )
            results.append(result)
    
    # Print summary
    print()
    print("=" * 60)
    print("🎉 Training Session Complete!")
    print(f"Finished: {datetime.now()}")
    print("=" * 60)
    print()
    
    # Summary table
    print("📊 Results Summary:")
    print("-" * 60)
    print(f"{'Model':<30} {'Status':<15} {'Duration':<15}")
    print("-" * 60)
    
    total_duration = 0
    success_count = 0
    
    for r in results:
        aug_str = "aug" if r["augment"] else "noaug"
        name = f"{r['model']}_{aug_str}"
        status_icon = "✅" if r["status"] == "completed" else "❌"
        print(f"{name:<30} {status_icon} {r['status']:<12} {format_duration(r['duration']):<15}")
        total_duration += r["duration"]
        if r["status"] == "completed":
            success_count += 1
    
    print("-" * 60)
    print(f"Total: {success_count}/{len(results)} completed in {format_duration(total_duration)}")
    print()
    
    # List outputs
    print("📂 Checkpoints:")
    for d in sorted(checkpoint_base.iterdir()):
        if d.is_dir():
            best_model = d / "best_model.pth"
            icon = "✅" if best_model.exists() else "❌"
            print(f"   {icon} {d.name}/")
    
    print()
    print("📦 Weights:")
    for f in sorted(weights_dir.glob("*.npy")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   ✅ {f.name} ({size_mb:.1f} MB)")
    
    # Save summary
    summary_file = log_dir / f"summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Training Session Summary\n")
        f.write(f"========================\n")
        f.write(f"Started: {timestamp}\n")
        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Total duration: {format_duration(total_duration)}\n\n")
        
        for r in results:
            f.write(f"{r['model']} (aug={r['augment']}): {r['status']} ({format_duration(r['duration'])})\n")
    
    print(f"\n📝 Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
