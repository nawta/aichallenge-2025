# TinyLidarNetBiLSTM

Bidirectional LSTM を用いた時系列モデル。学習時は未来情報も活用し、推論時は前方向のみで動作。

## アーキテクチャ

### 学習時 (Bidirectional)

```
入力:
  scans: (batch, seq_len, 1080)
  odoms: (batch, seq_len, 13)

[CNN Encoder - Per Frame]
  Conv1d(1→24→36→48→64→64) → Flatten → 1792
  FC(13→64) → 64
  Concat → 1856 → FC(1856→128) → 128

[Bidirectional LSTM]
  Input: (batch, seq_len, 128)
  Forward LSTM:  hidden_size=128 ──┐
  Backward LSTM: hidden_size=128 ──┼→ Concat → (batch, seq_len, 256)
                                   │
  Last timestep: (batch, 256) ←────┘

[Output Head]
  FC(256→64) + ReLU
  FC(64→2) + Tanh

出力: (batch, 2) - [acceleration, steering]
```

### 推論時 (Forward Only)

```
[CNN Encoder] → (128)
      │
      ▼
[Forward LSTM] → (128)  ← 隠れ状態を維持
      │
      ▼
[Projection Layer] → (256)  ← 次元を合わせる
      │
      ▼
[Output Head] → (2)
```

## Bidirectional LSTM の利点

```
      t-4    t-3    t-2    t-1     t
       │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼
  ┌─────────────────────────────────────┐
  │           Forward LSTM              │
  │   h₀ → h₁ → h₂ → h₃ → h₄           │
  └─────────────────────────────────────┘
                    +
  ┌─────────────────────────────────────┐
  │           Backward LSTM             │
  │   h₄ ← h₃ ← h₂ ← h₁ ← h₀           │
  └─────────────────────────────────────┘
                    │
                    ▼
            (256-dim output)
```

学習時は「未来」の情報も使って、より良い特徴を学習できる。

## 推論時の状態管理

```python
class BiLSTMInference:
    def __init__(self):
        self.h = None  # Hidden state
        self.c = None  # Cell state
    
    def process(self, scan, odom):
        # フレームをエンコード
        features = self.encode_frame(scan, odom)  # (1, 128)
        
        # Forward LSTM step
        output, (self.h, self.c) = self.lstm(features, (self.h, self.c))
        
        # 次元を揃えてHeadに入力
        projected = self.forward_proj(output)  # (1, 256)
        
        return self.head(projected)
    
    def reset_state(self):
        """エピソード開始時にリセット"""
        self.h = None
        self.c = None
```

## 他の時系列モデルとの比較

| 項目 | Stacked | BiLSTM | TCN |
|------|---------|--------|-----|
| 未来情報活用 | ❌ | ✅ (学習時) | ✅ (non-causal) |
| 長期依存 | 低 | **高** | 中〜高 |
| 可変長入力 | ❌ | **✅** | ❌ |
| リアルタイム推論 | 要バッファ | **状態維持** | 要バッファ |

## 使用方法

### 学習

```bash
python train.py \
  model.name='TinyLidarNetBiLSTM' \
  model.seq_len=10 \
  model.hidden_size=128
```

### 重み変換

```bash
python convert_weight.py \
  --model tinylidarnet_bilstm \
  --seq-len 10 \
  --hidden-size 128 \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/bilstm_weights.npy
```

### 推論設定

```yaml
model:
  architecture: "bilstm"
  seq_len: 10
  hidden_size: 128
  ckpt_path: "/path/to/bilstm_weights.npy"
```

## 特徴

- ✅ 長期依存関係のキャプチャ
- ✅ 学習時に未来情報を活用
- ✅ 可変長シーケンスに対応可能
- ✅ 状態維持でリアルタイム推論
- ❌ 学習時と推論時で挙動が異なる
- ❌ LSTM特有の勾配問題
- ❌ バッチ処理がStackedより複雑

## 推奨ユースケース

- 長期的な文脈が重要な制御
- 滑らかな出力が求められる場合
- 可変長のシーケンスを扱う場合
- リアルタイム推論（バッファ不要）

## パラメータ設定ガイド

| hidden_size | 用途 |
|-------------|------|
| 64 | 軽量、高速推論 |
| 128 | 推奨値、バランス良 |
| 256 | 高表現力、重い |

## 注意事項

### 学習時 vs 推論時の違い

```python
# 学習時
outputs = model(scans, odoms, use_bidirectional=True)

# 推論時
outputs = model(scans, odoms, use_bidirectional=False)
```

### 隠れ状態のリセット

エピソード開始時やシーンの切り替わり時に `reset_state()` を呼ぶ:

```python
# エピソード開始時
model.reset_state()

# シーン変更を検知した場合
if scene_changed:
    model.reset_state()
```
