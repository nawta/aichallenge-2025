# TinyLidarNet モデル一覧

このディレクトリには、TinyLidarNetの各モデルアーキテクチャの詳細ドキュメントを格納しています。

## モデル概要

### 単一フレームモデル

| モデル | 説明 | ドキュメント |
|--------|------|-------------|
| TinyLidarNet | 標準アーキテクチャ（5 Conv + 4 FC） | [tinylidarnet.md](./tinylidarnet.md) |
| TinyLidarNetSmall | 軽量版（3 Conv + 3 FC） | [tinylidarnet_small.md](./tinylidarnet_small.md) |
| TinyLidarNetDeep | 深層版（Residual + BatchNorm） | [tinylidarnet_deep.md](./tinylidarnet_deep.md) |
| TinyLidarNetFusion | odometry融合版 | [tinylidarnet_fusion.md](./tinylidarnet_fusion.md) |

### 時系列モデル

| モデル | 説明 | ドキュメント |
|--------|------|-------------|
| TinyLidarNetStacked | Frame Stacking | [tinylidarnet_stacked.md](./tinylidarnet_stacked.md) |
| TinyLidarNetBiLSTM | Bidirectional LSTM | [tinylidarnet_bilstm.md](./tinylidarnet_bilstm.md) |
| TinyLidarNetTCN | Temporal Convolutional Network | [tinylidarnet_tcn.md](./tinylidarnet_tcn.md) |

## モデル選択ガイド

```
推論速度重視 → TinyLidarNetSmall
↓
精度重視 → TinyLidarNetDeep
↓
車両状態を活用したい → TinyLidarNetFusion
↓
時系列情報を活用したい
├── シンプルに始めたい → TinyLidarNetStacked
├── 長期依存を重視 → TinyLidarNetBiLSTM
└── 高速 + 時系列 → TinyLidarNetTCN
```

## クイックスタート

### 学習

```bash
# 標準モデル
python train.py model.name='TinyLidarNet'

# 深層モデル
python train.py model.name='TinyLidarNetDeep'

# 時系列モデル
python train.py model.name='TinyLidarNetBiLSTM' model.seq_len=10
```

### 重み変換

```bash
python convert_weight.py \
  --model tinylidarnet \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/converted.npy
```

### 推論設定

```yaml
# tiny_lidar_net_node.param.yaml
model:
  architecture: "large"  # large, small, deep, fusion, stacked, bilstm, tcn
  ckpt_path: "/path/to/weights.npy"
```
