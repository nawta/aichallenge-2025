# TinyLidarNetDeep

深層アーキテクチャ。Residual Connections と BatchNorm を導入し、より高い表現力を実現。

## アーキテクチャ

```
入力: (batch, 1, 1080) - 1080点のLiDARスキャン

[Feature Extraction - Conv + BatchNorm]
  Conv1d(1→32, k=10, s=4) + BN + ReLU   → (batch, 32, 268)
  Conv1d(32→48, k=8, s=4) + BN + ReLU   → (batch, 48, 66)
  Conv1d(48→64, k=4, s=2) + BN + ReLU   → (batch, 64, 32)

[Residual Blocks]
  ResBlock1: Conv(64→64) + ReLU + Conv → Add(input) → ReLU
  ResBlock2: Conv(64→64) + ReLU + Conv → Add(input) → ReLU

[Additional Conv Layers]
  Conv1d(64→96, k=3) + BN + ReLU   → (batch, 96, 30)
  Conv1d(96→96, k=3) + BN + ReLU   → (batch, 96, 28)

[Flatten]
  96 × 28 = 2688

[Deep Regression Head]
  FC(2688→256) + ReLU + Dropout(0.2)
  FC(256→128) + ReLU
  FC(128→64) + ReLU + Dropout(0.2)
  FC(64→32) + ReLU
  FC(32→10) + ReLU
  FC(10→2) + Tanh

出力: (batch, 2) - [acceleration, steering]
```

## Residual Block

```
      ┌──────────────────────────┐
      │                          │
Input → Conv(k=3, p=1) → ReLU → Conv(k=3, p=1) → Add → ReLU → Output
      │                          ↑
      └──────────────────────────┘
              Skip Connection
```

勾配消失問題を緩和し、より深いネットワークの学習を可能にする。

## 標準版との比較

| 項目 | TinyLidarNet | TinyLidarNetDeep |
|-----|--------------|------------------|
| Conv層 | 5 | 5 + 4(res) |
| FC層 | 4 | 6 |
| Residual | ❌ | ✅ |
| BatchNorm | ❌ | ✅ |
| Dropout | ❌ | ✅ (0.2) |
| パラメータ数 | ~220K | ~750K |
| 表現力 | 中 | 高 |

## パラメータ数

| レイヤー | パラメータ数 |
|---------|-------------|
| conv1-3 + BN | ~20K |
| res_block1-2 | ~74K |
| conv4-5 + BN | ~19K |
| fc1-6 | ~640K |
| **合計** | **約750K** |

## 使用方法

### 学習

```bash
python train.py model.name='TinyLidarNetDeep'
```

### 重み変換

```bash
python convert_weight.py \
  --model tinylidarnet_deep \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/tinylidarnet_deep_weights.npy
```

### 推論設定

```yaml
model:
  architecture: "deep"
  ckpt_path: "/path/to/tinylidarnet_deep_weights.npy"
```

## 特徴

- ✅ 高い表現力
- ✅ Residual Connectionsで勾配消失を緩和
- ✅ BatchNormで学習の安定化
- ✅ Dropoutで過学習を抑制
- ❌ パラメータ数が多い
- ❌ 推論速度は標準版より遅い

## 推奨ユースケース

- 高精度が求められる場面
- 複雑なコースや環境
- 十分な計算リソースがある場合
- 学習データが豊富にある場合
