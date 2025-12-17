#!/usr/bin/env python3
"""
Quick validation script for all TinyLidarNet models.
Runs only a few batches per model to verify they work without full training.
"""
import sys
import traceback

import torch
import numpy as np

sys.path.insert(0, '/aichallenge/python_workspace/tiny_lidar_net')

from lib.model import (
    TinyLidarNet, TinyLidarNetSmall, TinyLidarNetDeep, TinyLidarNetFusion,
    TinyLidarNetStacked, TinyLidarNetBiLSTM, TinyLidarNetTCN, TinyLidarNetMap,
    TinyLidarNetLocalBEV, TinyLidarNetGlobalBEV, TinyLidarNetDualBEV
)


def test_model(model_name, model_class, kwargs, input_tensors, expected_output_shape):
    """Test a single model with forward pass."""
    try:
        model = model_class(**kwargs)
        model.eval()

        with torch.no_grad():
            output = model(*input_tensors)

        if output.shape != expected_output_shape:
            print(f"  Shape mismatch: got {output.shape}, expected {expected_output_shape}")
            return False

        # Check for NaN/Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"  Output contains NaN or Inf")
            return False

        print(f"  OK (output shape: {output.shape})")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Quick Model Validation for TinyLidarNet")
    print("=" * 60)

    batch_size = 4
    input_dim = 1080
    output_dim = 2
    state_dim = 13
    seq_len = 5
    hidden_size = 64
    map_feature_dim = 64
    local_grid_size = 32
    global_grid_size = 64

    device = "cpu"

    passed = 0
    failed = 0

    # Generate test data
    scan = torch.randn(batch_size, input_dim, device=device)
    state = torch.randn(batch_size, state_dim, device=device)
    scan_seq = torch.randn(batch_size, seq_len, input_dim, device=device)
    state_seq = torch.randn(batch_size, seq_len, state_dim, device=device)
    local_bev = torch.randn(batch_size, 2, local_grid_size, local_grid_size, device=device)
    global_bev = torch.randn(batch_size, 3, global_grid_size, global_grid_size, device=device)

    tests = [
        # Single-frame models
        ("TinyLidarNet", TinyLidarNet,
         {"input_dim": input_dim, "output_dim": output_dim},
         (scan,), torch.Size([batch_size, output_dim])),

        ("TinyLidarNetSmall", TinyLidarNetSmall,
         {"input_dim": input_dim, "output_dim": output_dim},
         (scan,), torch.Size([batch_size, output_dim])),

        ("TinyLidarNetDeep", TinyLidarNetDeep,
         {"input_dim": input_dim, "output_dim": output_dim},
         (scan,), torch.Size([batch_size, output_dim])),

        ("TinyLidarNetFusion", TinyLidarNetFusion,
         {"input_dim": input_dim, "state_dim": state_dim, "output_dim": output_dim},
         (scan, state), torch.Size([batch_size, output_dim])),

        # Temporal models
        ("TinyLidarNetStacked", TinyLidarNetStacked,
         {"input_dim": input_dim, "state_dim": state_dim, "seq_len": seq_len, "output_dim": output_dim},
         (scan_seq, state_seq), torch.Size([batch_size, output_dim])),

        ("TinyLidarNetBiLSTM", TinyLidarNetBiLSTM,
         {"input_dim": input_dim, "state_dim": state_dim, "hidden_size": hidden_size, "output_dim": output_dim},
         (scan_seq, state_seq), torch.Size([batch_size, output_dim])),

        ("TinyLidarNetTCN", TinyLidarNetTCN,
         {"input_dim": input_dim, "state_dim": state_dim, "hidden_size": hidden_size,
          "causal": False, "output_dim": output_dim},
         (scan_seq, state_seq), torch.Size([batch_size, output_dim])),

        # Map model (without actual map - just test forward pass shapes)
        ("TinyLidarNetMap", TinyLidarNetMap,
         {"input_dim": input_dim, "map_feature_dim": map_feature_dim, "output_dim": output_dim},
         (scan,), torch.Size([batch_size, output_dim])),

        # BEV models
        ("TinyLidarNetLocalBEV", TinyLidarNetLocalBEV,
         {"input_dim": input_dim, "local_grid_size": local_grid_size, "local_channels": 2,
          "state_dim": state_dim, "output_dim": output_dim},
         (scan, local_bev, state), torch.Size([batch_size, output_dim])),

        ("TinyLidarNetGlobalBEV", TinyLidarNetGlobalBEV,
         {"input_dim": input_dim, "global_grid_size": global_grid_size, "global_channels": 3,
          "state_dim": state_dim, "output_dim": output_dim},
         (scan, global_bev, state), torch.Size([batch_size, output_dim])),

        ("TinyLidarNetDualBEV", TinyLidarNetDualBEV,
         {"input_dim": input_dim, "local_grid_size": local_grid_size, "local_channels": 2,
          "global_grid_size": global_grid_size, "global_channels": 3,
          "state_dim": state_dim, "output_dim": output_dim},
         (scan, local_bev, global_bev, state), torch.Size([batch_size, output_dim])),
    ]

    for name, model_class, kwargs, inputs, expected_shape in tests:
        print(f"\nTesting {name}...")
        if test_model(name, model_class, kwargs, inputs, expected_shape):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
