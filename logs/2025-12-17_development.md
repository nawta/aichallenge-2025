# Development Log - 2025-12-17

## Overview
TinyLidarNetモデルの改善作業を実施。より深いネットワークアーキテクチャを追加し、ゴール到達率の向上を目指す。

---

## 🔍 問題分析

### 現状の課題
- TinyLidarNetモデルがゴールに到達しない
- 学習データと推論環境のドメインギャップの可能性
- `max_range` の設定不一致（laserscan_generator: 100m vs model: 30m）

### 原因の可能性
1. モデルの表現力不足
2. 入力データの正規化ミスマッチ
3. アクセル制御の学習困難

---

## ✅ 実施した変更

### 1. TinyLidarNetDeep モデルの追加

#### アーキテクチャ概要
```
入力: (batch, 1, 1080) - 1080点のLiDARスキャン

[Feature Extraction - Conv + BatchNorm]
  Conv1d(1→32, k=10, s=4) + BN + ReLU  → 268
  Conv1d(32→48, k=8, s=4) + BN + ReLU  → 66
  Conv1d(48→64, k=4, s=2) + BN + ReLU  → 32

[Residual Blocks - Skip Connections]
  ResBlock1: Conv(64→64, k=3, p=1) → ReLU → Conv → Add(input) → ReLU
  ResBlock2: Conv(64→64, k=3, p=1) → ReLU → Conv → Add(input) → ReLU

[Additional Conv Layers]
  Conv1d(64→96, k=3) + BN + ReLU  → 30
  Conv1d(96→96, k=3) + BN + ReLU  → 28

[Regression Head - FC + Dropout]
  Flatten → 2688
  FC(2688→256) + ReLU + Dropout(0.2)
  FC(256→128) + ReLU
  FC(128→64) + ReLU + Dropout(0.2)
  FC(64→32) + ReLU
  FC(32→10) + ReLU
  FC(10→2) + Tanh

出力: (batch, 2) - [acceleration, steering]
```

#### 新機能
- **Residual Connections**: 勾配消失問題の緩和
- **Batch Normalization**: 学習の安定化
- **Dropout (0.2)**: 過学習の抑制
- **より広いFC層**: 256→128→64→32→10→2

### 2. 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `python_workspace/tiny_lidar_net/lib/model.py` | `ResidualBlock1d`, `TinyLidarNetDeep` クラス追加 |
| `tiny_lidar_net_controller/.../tinylidarnet.py` | `TinyLidarNetDeep`, `TinyLidarNetDeepNp` クラス追加 |
| `tiny_lidar_net_controller/.../numpy/layers.py` | `conv1d_padded`, `batch_norm1d` 関数追加 |
| `tiny_lidar_net_controller/.../numpy/initializers.py` | `ones_init` 関数追加 |
| `tiny_lidar_net_controller/.../model/__init__.py` | 新関数のエクスポート追加 |
| `python_workspace/tiny_lidar_net/convert_weight.py` | `tinylidarnet_deep` オプション追加 |
| `python_workspace/tiny_lidar_net/train.py` | `TinyLidarNetDeep` モデル選択追加 |
| `python_workspace/tiny_lidar_net/config/train.yaml` | モデル名コメント更新 |
| `tiny_lidar_net_controller/.../tiny_lidar_net_controller_core.py` | `deep` アーキテクチャサポート追加 |
| `tiny_lidar_net_controller/config/tiny_lidar_net_node.param.yaml` | `architecture` オプション更新 |

---

## 📊 モデル比較

| モデル | Conv層 | FC層 | Residual | BatchNorm | Dropout | 推定パラメータ数 |
|--------|-------|------|----------|-----------|---------|-----------------|
| TinyLidarNet | 5 | 4 | ❌ | ❌ | ❌ | ~251K |
| TinyLidarNetSmall | 3 | 3 | ❌ | ❌ | ❌ | ~213K |
| **TinyLidarNetDeep** | 5+4(res) | 6 | ✅ | ✅ | ✅ | **~750K** |

---

## 🚀 使用方法

### 学習
```bash
# TinyLidarNetDeep で学習
python3 train.py model.name='TinyLidarNetDeep'

# ステアのみ学習（推奨）
python3 train.py \
  model.name='TinyLidarNetDeep' \
  loss.steer_weight=1.0 \
  loss.accel_weight=0.0
```

### 重み変換 (PyTorch → NumPy)
```bash
python3 convert_weight.py \
  --model tinylidarnet_deep \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/tinylidarnet_deep_weights.npy
```

### 推論設定 (tiny_lidar_net_node.param.yaml)
```yaml
model:
  architecture: "deep"  # "large", "small", "deep" から選択
  ckpt_path: "/path/to/tinylidarnet_deep_weights.npy"
```

---

## 📝 今後の改善案

### 短期
- [ ] `max_range` を 30m に統一
- [ ] `control_mode: fixed` でアクセル固定、ステアのみAI予測
- [ ] デバッグモードで推論出力を確認

### 中期
- [ ] laserscan_generator 環境で学習データを再収集
- [ ] コーナーや難所でのデータ増量
- [ ] 学習率スケジューラーの導入

### 長期
- [ ] Data Augmentation の追加（ノイズ、回転など）
- [ ] Attention機構の導入検討
- [ ] 速度に応じた制御戦略の分岐

---

## 🐛 既知の問題

1. **ドメインギャップ**: 学習データ（rosbag）と推論環境（laserscan_generator）の違い
2. **アクセル学習困難**: README記載の通り、アクセルの学習がうまくいかない傾向

---

## 📚 参考

- [TinyLidarNet Paper (arXiv:2410.07447)](https://arxiv.org/abs/2410.07447)
- [CSL-KU/TinyLidarNet GitHub](https://github.com/CSL-KU/TinyLidarNet)
