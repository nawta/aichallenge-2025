# TinyLidarNetSmall

軽量版アーキテクチャ。推論速度を優先する場合に使用。

## アーキテクチャ

```
入力: (batch, 1, 1080) - 1080点のLiDARスキャン

[Convolutional Layers]
  Conv1d(1→24, k=10, s=4) + ReLU   → (batch, 24, 268)
  Conv1d(24→36, k=8, s=4) + ReLU   → (batch, 36, 66)
  Conv1d(36→48, k=4, s=2) + ReLU   → (batch, 48, 32)

[Flatten]
  48 × 32 = 1536

[Fully Connected Layers]
  FC(1536→100) + ReLU
  FC(100→50) + ReLU
  FC(50→2) + Tanh

出力: (batch, 2) - [acceleration, steering]
```

## 標準版との比較

| 項目 | TinyLidarNet | TinyLidarNetSmall |
|-----|--------------|-------------------|
| Conv層 | 5 | 3 |
| FC層 | 4 | 3 |
| パラメータ数 | ~220K | ~165K |
| 推論速度 | 速い | より速い |
| 表現力 | 中 | 低 |

## パラメータ数

| レイヤー | パラメータ数 |
|---------|-------------|
| conv1 | 24 × 1 × 10 + 24 = 264 |
| conv2 | 36 × 24 × 8 + 36 = 6,948 |
| conv3 | 48 × 36 × 4 + 48 = 6,960 |
| fc1 | 1536 × 100 + 100 = 153,700 |
| fc2 | 100 × 50 + 50 = 5,050 |
| fc3 | 50 × 2 + 2 = 102 |
| **合計** | **約173K** |

## 使用方法

### 学習

```bash
python train.py model.name='TinyLidarNetSmall'
```

### 重み変換

```bash
python convert_weight.py \
  --model tinylidarnet_small \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/tinylidarnet_small_weights.npy
```

### 推論設定

```yaml
model:
  architecture: "small"
  ckpt_path: "/path/to/tinylidarnet_small_weights.npy"
```

## 特徴

- ✅ 最も高速な推論
- ✅ 最小のパラメータ数
- ✅ メモリ効率が良い
- ❌ 表現力が限られる
- ❌ 複雑なシーンでの精度低下の可能性

## 推奨ユースケース

- リソース制約のある環境
- リアルタイム性が最優先の場合
- ベースラインとしての使用
- 軽量デバイスでの実行
