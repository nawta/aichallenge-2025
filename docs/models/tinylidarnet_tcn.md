# TinyLidarNetTCN

Temporal Convolutional Network (TCN) を用いた時系列モデル。Dilated Causal Convolution で効率的に長期依存をキャプチャ。

## アーキテクチャ

```
入力:
  scans: (batch, seq_len, 1080)
  odoms: (batch, seq_len, 13)

[CNN Encoder - Per Frame]
  Conv1d(1→24→36→48→64→64) → Flatten → 1792
  FC(13→64) → 64
  Concat → 1856 → FC(1856→128) → 128

[TCN Blocks]
  Input: (batch, 128, seq_len)  ← 時間軸が最後
  
  TemporalBlock(d=1): Conv(k=3, dilation=1) × 2 + Residual
  TemporalBlock(d=2): Conv(k=3, dilation=2) × 2 + Residual
  TemporalBlock(d=4): Conv(k=3, dilation=4) × 2 + Residual

  Last timestep: (batch, 128)

[Output Head]
  FC(128→64) + ReLU
  FC(64→2) + Tanh

出力: (batch, 2) - [acceleration, steering]
```

## Temporal Block

```
         ┌─────────────────────────────────────────┐
         │              Residual                   │
         │                                         │
Input ───┼──→ Causal Conv(d) → ReLU → Causal Conv(d) → ReLU → Add → Output
         │                                         ↑
         └─────────────────────────────────────────┘
```

## Dilated Causal Convolution

```
Dilation = 1:
  ○─○─●    (receptive field = 3)

Dilation = 2:
  ○───○───●    (receptive field = 5)

Dilation = 4:
  ○───────○───────●    (receptive field = 9)

Combined (d=1,2,4):
  ○───○───○───○───○───○───○───○───●
  (receptive field = 15 with k=3, 3 levels)
```

**Causal Convolution**: 未来の情報を見ない（推論時安全）

```
標準Convolution:        Causal Convolution:
  [past][now][future]     [past][past][now]
        ↓                       ↓
      output                  output
```

## 他の時系列モデルとの比較

| 項目 | Stacked | BiLSTM | TCN |
|------|---------|--------|-----|
| 並列計算 | ✅ | ❌ | **✅** |
| 長期依存 | 低 | 高 | **中〜高** |
| 勾配安定性 | 高 | 中 | **高** |
| 推論速度 | 高速 | 中 | **高速** |
| Receptive Field | seq_len | 無限 | 設定可能 |

## Receptive Field の計算

```
receptive_field = 1 + Σ(kernel_size - 1) × dilation
                = 1 + 2 × (1 + 2 + 4)  (3 levels, k=3)
                = 1 + 2 × 7
                = 15
```

| num_levels | dilations | receptive_field (k=3) |
|------------|-----------|----------------------|
| 2 | 1, 2 | 7 |
| 3 | 1, 2, 4 | 15 |
| 4 | 1, 2, 4, 8 | 31 |

## 使用方法

### 学習

```bash
# Causal (推論と同じ)
python train.py \
  model.name='TinyLidarNetTCN' \
  model.seq_len=10 \
  model.hidden_size=128 \
  model.tcn_causal=true

# Non-causal (未来情報を活用)
python train.py \
  model.name='TinyLidarNetTCN' \
  model.tcn_causal=false
```

### 重み変換

```bash
python convert_weight.py \
  --model tinylidarnet_tcn \
  --seq-len 10 \
  --hidden-size 128 \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/tcn_weights.npy
```

### 推論設定

```yaml
model:
  architecture: "tcn"
  seq_len: 10
  hidden_size: 128
  ckpt_path: "/path/to/tcn_weights.npy"
```

## 推論時のバッファ管理

```python
from collections import deque

class TCNInference:
    def __init__(self, seq_len=10):
        self.scan_buffer = deque(maxlen=seq_len)
        self.odom_buffer = deque(maxlen=seq_len)
    
    def process(self, scan, odom):
        self.scan_buffer.append(scan)
        self.odom_buffer.append(odom)
        
        if len(self.scan_buffer) < seq_len:
            return None  # Not enough frames
        
        # 全シーケンスを処理
        stacked_scans = np.stack(list(self.scan_buffer))
        stacked_odoms = np.stack(list(self.odom_buffer))
        return self.model(stacked_scans, stacked_odoms)
```

## 特徴

- ✅ 高速な並列計算
- ✅ 安定した勾配（勾配消失しにくい）
- ✅ Receptive Fieldを明示的に制御可能
- ✅ 学習が安定
- ❌ フレームバッファが必要
- ❌ 可変長シーケンスには不向き

## 推奨ユースケース

- 高速な推論が求められる場合
- 長期依存 + 速度のバランス
- 学習の安定性が重要な場合
- GPUでの並列処理を活用したい場合

## パラメータ設定ガイド

| パラメータ | 推奨値 | 説明 |
|-----------|-------|------|
| hidden_size | 128 | TCN隠れ層の次元 |
| num_levels | 3 | TCNブロック数 |
| kernel_size | 3 | 畳み込みカーネルサイズ |
| causal | true | 推論用はtrue推奨 |

### Causal vs Non-causal

| モード | 学習 | 推論 | 用途 |
|--------|------|------|------|
| causal=true | 過去のみ | 過去のみ | 一貫した動作 |
| causal=false | 過去+未来 | 過去のみ* | より良い学習* |

*Non-causal学習後、推論時はcausalで動作（パディング調整）
