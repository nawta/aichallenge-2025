# TinyLidarNetStacked

Frame Stacking による時系列モデル。過去N フレームをチャンネルとして結合するシンプルなアプローチ。

## アーキテクチャ

```
入力:
  scans: (batch, seq_len, 1080) - 連続LiDARスキャン
  odoms: (batch, seq_len, 13) - 連続odometry

[LiDAR Branch - Stacked Input]
  Conv1d(in_ch=seq_len, 24, k=10, s=4) + ReLU   → 268
  Conv1d(24→36, k=8, s=4) + ReLU               → 66
  Conv1d(36→48, k=4, s=2) + ReLU               → 32
  Conv1d(48→64, k=3, s=1) + ReLU               → 30
  Conv1d(64→64, k=3, s=1) + ReLU               → 28
  Flatten → 1792

[State Branch - Flattened Sequence]
  Flatten(seq_len × 13) → FC(130→64) + ReLU → 64

[Fusion]
  Concat(1792, 64) → 1856

[Regression Head]
  FC(1856→100) + ReLU
  FC(100→50) + ReLU
  FC(50→10) + ReLU
  FC(10→2) + Tanh

出力: (batch, 2) - [acceleration, steering]
```

## Frame Stacking の概念

```
時刻:  t-9  t-8  t-7  t-6  t-5  t-4  t-3  t-2  t-1   t
        │    │    │    │    │    │    │    │    │    │
        ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
      ┌────────────────────────────────────────────────┐
      │     10フレームをチャンネル方向にスタック         │
      │     (seq_len=10, scan_dim=1080)                │
      └────────────────────────────────────────────────┘
                           │
                           ▼
               Conv1d(in_channels=10, ...)
```

## 他の時系列モデルとの比較

| 項目 | Stacked | BiLSTM | TCN |
|------|---------|--------|-----|
| 実装難易度 | 低 | 中 | 中 |
| 推論速度 | 高速 | 中 | 高速 |
| 長期依存 | 低 | 高 | 中〜高 |
| パラメータ効率 | 低 | 高 | 中 |
| 可変長入力 | ❌ | ✅ | ❌ |

## 使用方法

### 学習

```bash
python train.py \
  model.name='TinyLidarNetStacked' \
  model.seq_len=10
```

### 重み変換

```bash
python convert_weight.py \
  --model tinylidarnet_stacked \
  --seq-len 10 \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/stacked_weights.npy
```

### 推論設定

```yaml
model:
  architecture: "stacked"
  seq_len: 10
  ckpt_path: "/path/to/stacked_weights.npy"
```

## 推論時のバッファ管理

```python
from collections import deque

class StackedInference:
    def __init__(self, seq_len=10):
        self.scan_buffer = deque(maxlen=seq_len)
        self.odom_buffer = deque(maxlen=seq_len)
    
    def process(self, scan, odom):
        self.scan_buffer.append(scan)
        self.odom_buffer.append(odom)
        
        # バッファが満たされるまで待機
        if len(self.scan_buffer) < seq_len:
            return None
        
        # スタックして推論
        stacked_scans = np.stack(list(self.scan_buffer))
        stacked_odoms = np.stack(list(self.odom_buffer))
        return self.model(stacked_scans, stacked_odoms)
```

## 特徴

- ✅ 実装がシンプル
- ✅ 推論が高速
- ✅ 時系列の空間的パターンを学習
- ❌ seq_len固定（可変長不可）
- ❌ 長期依存のキャプチャが困難
- ❌ パラメータ効率が悪い（seq_len分のチャンネル）

## 推奨ユースケース

- 時系列モデルの初期検証
- シンプルなベースライン
- 短い時間窓（5-15フレーム）で十分な場合
- 推論速度が重要な場合

## パラメータ設定ガイド

| seq_len | 時間窓 (10Hz) | 用途 |
|---------|--------------|------|
| 5 | 0.5秒 | 短期的な変化 |
| 10 | 1.0秒 | 推奨値 |
| 20 | 2.0秒 | より長い履歴 |
