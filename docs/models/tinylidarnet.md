# TinyLidarNet

標準的なCNNアーキテクチャ。オリジナルの[TinyLidarNet論文](https://arxiv.org/abs/2410.07447)に基づく実装。

## アーキテクチャ

```
入力: (batch, 1, 1080) - 1080点のLiDARスキャン

[Convolutional Layers]
  Conv1d(1→24, k=10, s=4) + ReLU   → (batch, 24, 268)
  Conv1d(24→36, k=8, s=4) + ReLU   → (batch, 36, 66)
  Conv1d(36→48, k=4, s=2) + ReLU   → (batch, 48, 32)
  Conv1d(48→64, k=3, s=1) + ReLU   → (batch, 64, 30)
  Conv1d(64→64, k=3, s=1) + ReLU   → (batch, 64, 28)

[Flatten]
  64 × 28 = 1792

[Fully Connected Layers]
  FC(1792→100) + ReLU
  FC(100→50) + ReLU
  FC(50→10) + ReLU
  FC(10→2) + Tanh

出力: (batch, 2) - [acceleration, steering]
```

## パラメータ数

| レイヤー | パラメータ数 |
|---------|-------------|
| conv1 | 24 × 1 × 10 + 24 = 264 |
| conv2 | 36 × 24 × 8 + 36 = 6,948 |
| conv3 | 48 × 36 × 4 + 48 = 6,960 |
| conv4 | 64 × 48 × 3 + 64 = 9,280 |
| conv5 | 64 × 64 × 3 + 64 = 12,352 |
| fc1 | 1792 × 100 + 100 = 179,300 |
| fc2 | 100 × 50 + 50 = 5,050 |
| fc3 | 50 × 10 + 10 = 510 |
| fc4 | 10 × 2 + 2 = 22 |
| **合計** | **約220K** |

## 使用方法

### 学習

```bash
python train.py model.name='TinyLidarNet'
```

### 重み変換

```bash
python convert_weight.py \
  --model tinylidarnet \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/tinylidarnet_weights.npy
```

### 推論設定

```yaml
model:
  architecture: "large"
  ckpt_path: "/path/to/tinylidarnet_weights.npy"
```

## 特徴

- ✅ シンプルな構造で理解しやすい
- ✅ 推論速度が速い
- ✅ 少ないパラメータ数
- ❌ 表現力に限界がある
- ❌ 時系列情報を活用しない

## 入出力仕様

### 入力

| 名前 | 形状 | 説明 |
|-----|------|------|
| scan | (batch, 1, 1080) | 正規化済みLiDARスキャン（0〜1） |

### 出力

| 名前 | 形状 | 範囲 | 説明 |
|-----|------|------|------|
| acceleration | (batch,) | [-1, 1] | 加速度指令 |
| steering | (batch,) | [-1, 1] | ステアリング角度 |

## 参考

- [TinyLidarNet Paper (arXiv:2410.07447)](https://arxiv.org/abs/2410.07447)
- [CSL-KU/TinyLidarNet GitHub](https://github.com/CSL-KU/TinyLidarNet)
