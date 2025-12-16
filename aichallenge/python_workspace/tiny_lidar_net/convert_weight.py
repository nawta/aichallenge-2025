import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import torch

from lib.model import (
    TinyLidarNet, TinyLidarNetSmall, TinyLidarNetDeep, TinyLidarNetFusion,
    TinyLidarNetStacked, TinyLidarNetBiLSTM, TinyLidarNetTCN
)


def extract_params_to_dict(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    """Extracts the state dictionary from a PyTorch model and converts it to a NumPy dictionary.

    This function acts as a pure transformation layer, isolating the logic of
    parameter extraction and naming convention changes (dot to underscore) from
    file I/O operations. This design ensures high testability.

    Args:
        model: The PyTorch model instance to extract weights from.

    Returns:
        A dictionary mapping parameter names (with underscores replaced) to
        detached NumPy arrays on the CPU.
    """
    return {
        k.replace('.', '_'): v.detach().cpu().numpy()
        for k, v in model.state_dict().items()
    }


def save_numpy_dict(params: Dict[str, np.ndarray], output_path: Path) -> None:
    """Saves a NumPy dictionary to a file system path.

    Handles the creation of parent directories if they do not exist and
    persists the parameter dictionary as a .npy file.

    Args:
        params: The dictionary of model parameters.
        output_path: The filesystem path where the .npy file will be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, params)
    print(f"Saved NumPy weights to: {output_path}")


def load_model(
    model_name: str, input_dim: int, output_dim: int, ckpt_path: Path,
    state_dim: int = 13, seq_len: int = 10, hidden_size: int = 128
) -> torch.nn.Module:
    """Initializes the model architecture and loads weights from a checkpoint.

    Args:
        model_name: The name of the architecture.
        input_dim: The size of the input dimension (e.g., LiDAR rays).
        output_dim: The size of the output dimension (e.g., control commands).
        ckpt_path: The path to the PyTorch checkpoint file (.pth).
        state_dim: The size of the state dimension (default: 13).
        seq_len: Sequence length for temporal models (default: 10).
        hidden_size: Hidden size for temporal models (default: 128).

    Returns:
        The PyTorch model instance with loaded weights.

    Raises:
        ValueError: If the provided model_name is not supported.
        FileNotFoundError: If the checkpoint file does not exist at ckpt_path.
    """
    if model_name == "tinylidarnet":
        model = TinyLidarNet(input_dim=input_dim, output_dim=output_dim)
    elif model_name == "tinylidarnet_small":
        model = TinyLidarNetSmall(input_dim=input_dim, output_dim=output_dim)
    elif model_name == "tinylidarnet_deep":
        model = TinyLidarNetDeep(input_dim=input_dim, output_dim=output_dim)
    elif model_name == "tinylidarnet_fusion":
        model = TinyLidarNetFusion(
            input_dim=input_dim, 
            state_dim=state_dim,
            output_dim=output_dim
        )
    elif model_name == "tinylidarnet_stacked":
        model = TinyLidarNetStacked(
            input_dim=input_dim,
            state_dim=state_dim,
            seq_len=seq_len,
            output_dim=output_dim
        )
    elif model_name == "tinylidarnet_bilstm":
        model = TinyLidarNetBiLSTM(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_size=hidden_size,
            output_dim=output_dim
        )
    elif model_name == "tinylidarnet_tcn":
        model = TinyLidarNetTCN(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_size=hidden_size,
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def convert_checkpoint(
    model_name: str, input_dim: int, output_dim: int, ckpt: Path, output: Path,
    state_dim: int = 13, seq_len: int = 10, hidden_size: int = 128
) -> None:
    """Orchestrates the model conversion process.

    This function combines the loading of the model architecture, the extraction
    of parameters into a pure dictionary format, and the saving to disk.

    Args:
        model_name: The name of the architecture to load.
        input_dim: The input dimension size.
        output_dim: The output dimension size.
        ckpt: The source path to the PyTorch checkpoint.
        output: The destination path for the converted NumPy file.
        state_dim: The state dimension size.
        seq_len: Sequence length for temporal models.
        hidden_size: Hidden size for temporal models.
    """
    # 1. Load Model (I/O & Logic)
    model = load_model(
        model_name, input_dim, output_dim, ckpt, 
        state_dim=state_dim, seq_len=seq_len, hidden_size=hidden_size
    )
    
    # 2. Extract Parameters (Pure Logic) -> Easy to Unit Test
    params = extract_params_to_dict(model)
    
    # 3. Save to Disk (I/O)
    save_numpy_dict(params, output)


def main() -> None:
    """Main entry point for the command-line interface.

    Parses command-line arguments and triggers the checkpoint conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert PyTorch weights to NumPy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, 
                        choices=[
                            "tinylidarnet", "tinylidarnet_small", "tinylidarnet_deep", 
                            "tinylidarnet_fusion", "tinylidarnet_stacked", 
                            "tinylidarnet_bilstm", "tinylidarnet_tcn"
                        ], 
                        default="tinylidarnet", help="Model architecture")
    parser.add_argument("--input-dim", type=int, default=1080, help="Input dimension size")
    parser.add_argument("--output-dim", type=int, default=2, help="Output dimension size")
    parser.add_argument("--state-dim", type=int, default=13, help="State dimension size")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length (for temporal models)")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size (for temporal models)")
    parser.add_argument("--ckpt", type=Path, required=True, help="Source .pth checkpoint")
    parser.add_argument("--output", type=Path, default=Path("./weights/converted_weights.npy"), help="Destination .npy path")

    args = parser.parse_args()

    convert_checkpoint(
        args.model, args.input_dim, args.output_dim, args.ckpt, args.output,
        state_dim=args.state_dim, seq_len=args.seq_len, hidden_size=args.hidden_size
    )


if __name__ == "__main__":
    main()
