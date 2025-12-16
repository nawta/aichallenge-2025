# Development Log - 2025-12-17

## Overview
TinyLidarNetモデルの大幅な改善作業を実施。以下の機能を追加：

1. **TinyLidarNetDeep**: より深いネットワークアーキテクチャ（Residual + BatchNorm）
2. **Rosbagクロップ**: 学習データの前後5秒をカットして品質向上
3. **Data Augmentation**: Mirror Augmentation（左右反転）の実装
4. **TinyLidarNetFusion**: kinematic state（odometry）との融合モデル
5. **Temporal Models**: 時系列を考慮した3つの新モデル
   - TinyLidarNetStacked（Frame Stacking）
   - TinyLidarNetBiLSTM（Bidirectional LSTM）
   - TinyLidarNetTCN（Temporal Convolutional Network）

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
- [ ] 各モデルの比較実験を実施

### 中期
- [ ] laserscan_generator 環境で学習データを再収集
- [ ] コーナーや難所でのデータ増量
- [ ] 学習率スケジューラーの導入
- [ ] ハイパーパラメータチューニング（seq_len, hidden_size等）

### 長期
- [x] Data Augmentation の追加（Mirror実装済み）
- [x] 時系列モデルの実装（Stacked, BiLSTM, TCN）
- [ ] Attention機構の導入検討
- [ ] 速度に応じた制御戦略の分岐

---

## 🗂️ Rosbag データセットのクロップ

### 背景
学習データ（rosbag）の最初と最後の5秒間は、車両が停止中または不安定な状態のデータが含まれている可能性がある。これらを除去することでデータ品質を向上させる。

### 実施内容

#### 1. ros2bag_extensions のインストール
```bash
# tier4/ros2bag_extensions を使用
git clone https://github.com/tier4/ros2bag_extensions.git
colcon build
```

#### 2. クロップスクリプトの作成
`aichallenge/crop_rosbags.py` を作成:
- zstd圧縮されたMCAPファイルに対応
- 各rosbagの最初5秒と最後5秒を自動カット
- `--replace`: オリジナルをクロップ版に置換
- `--cleanup`: バックアップ（オリジナル）を削除

#### 3. データセットの処理結果

| データセット | 処理数 | 備考 |
|-------------|-------|------|
| train | 21 bags | 全てクロップ完了 |
| val | 5 bags | 全てクロップ完了 |

#### 4. 例：train0 の変更
```
Original: 77.4s (12:51:57 〜 12:53:15)
Cropped:  67.4s (12:52:02 〜 12:53:10)
→ 10秒短縮（前後5秒ずつカット）
```

### 使用方法

```bash
# Docker コンテナ内で実行
cd /aichallenge

# クロップ実行（_cropped サフィックスで保存）
python3 crop_rosbags.py

# オリジナルをクロップ版に置換
python3 crop_rosbags.py --replace

# バックアップを削除
python3 crop_rosbags.py --cleanup
```

### ファイル構成
```
/aichallenge/dataset/
├── train/
│   ├── rosbag2_autoware_man_train0/      # クロップ版
│   ├── rosbag2_autoware_man_train1/
│   └── ...
├── val/
│   ├── rosbag2_autoware_man_val0/
│   └── ...
└── ...

/aichallenge/dataset_backup/              # オリジナル（バックアップ）
├── train/
│   ├── rosbag2_autoware_man_train0_original/
│   └── ...
└── val/
    └── ...
```

---

## 🐛 既知の問題

1. **ドメインギャップ**: 学習データ（rosbag）と推論環境（laserscan_generator）の違い
2. **アクセル学習困難**: README記載の通り、アクセルの学習がうまくいかない傾向

---

## 🔄 Data Augmentation の実装

### 背景
学習データを効果的に増やすため、オンライン（学習時動的）のデータ拡張機能を追加。

### Mirror Augmentation（左右反転）

#### 仕組み
```
元データ:                          反転後:
LiDARスキャン [左→右]    →    LiDARスキャン [右→左]
steer = +0.1             →    steer = -0.1
accel = 0.5              →    accel = 0.5 (変化なし)
```

#### 実装
- `ScanControlSequenceDataset.__getitem__()` でオンライン適用
- `augment_mirror`: 有効/無効の切り替え（**デフォルト: ON**）
- `augment_prob`: 適用確率（デフォルト: 0.5 = 50%）

#### 設定方法 (config/train.yaml)
```yaml
data:
  augment_mirror: true   # false で無効化
  augment_prob: 0.5      # 適用確率
```

#### 重要: Temporal モデルとの整合性
テンポラル情報を扱うモデル（Frame Stacking, LSTM, TCN等）では、フレームごとに独立して反転すると時系列の一貫性が壊れる。

**解決策**: シーケンス単位で1回だけ反転判定を行い、シーケンス内の全フレームを同時に反転：

```python
def __getitem__(self, idx):
    # シーケンス全体で1回だけ判定
    apply_mirror = self.augment_mirror and np.random.random() < self.augment_prob
    
    if self.seq_len > 1:
        scans = self.scans[idx:end_idx]  # (seq_len, scan_dim)
        if apply_mirror:
            # axis=1 (scan_dim) を反転、axis=0 (時間軸) は保持
            scans = np.flip(scans, axis=1).copy()
            steer = -steer
```

---

## 🕐 Temporal Models の追加

### 背景
LiDARデータは時系列データであり、連続するフレーム間の関係を利用することで、より正確な予測が可能になる可能性がある。約10フレーム（約1秒）の履歴を考慮する3つの時系列モデルを実装。

### 追加モデル一覧

| モデル名 | アーキテクチャ | 特徴 | 推論速度 |
|---------|--------------|------|---------|
| `TinyLidarNetStacked` | Frame Stacking | 複数フレームをチャンネルとして結合 | 高速 |
| `TinyLidarNetBiLSTM` | Bidirectional LSTM | 学習時は未来も見る、推論時は前方向のみ | 中 |
| `TinyLidarNetTCN` | Temporal Conv Network | Dilated Causal Convolution | 高速 |

### アーキテクチャ詳細

#### 1. TinyLidarNetStacked (Frame Stacking)
```
入力: scans (batch, seq_len, scan_dim), odoms (batch, seq_len, 13)

[LiDAR Branch]
  Conv1d(in_ch=seq_len, 24, k=10, s=4)  # seq_lenフレームをチャンネルとして
  → Conv2〜5 (same as TinyLidarNet)
  → Flatten → 1792

[Odom Branch]
  Flatten(seq_len * 13) → FC(130→64)

[Fusion]
  Concat(1792 + 64) → FC Head → Output (2)
```

#### 2. TinyLidarNetBiLSTM (Bidirectional LSTM)
```
学習時:
  [CNN Encoder] → [BiLSTM] → [FC Head]
       ↓              ↓
    (128-dim)    (256-dim)  ← Forward + Backward

推論時:
  [CNN Encoder] → [Forward LSTM] → [Projection] → [FC Head]
       ↓              ↓               ↓
    (128-dim)     (128-dim)       (256-dim)
```

**設計ポイント**:
- 学習時は双方向LSTM（未来情報も活用）
- 推論時は前方向LSTMのみ + Projection層で次元を合わせる
- 隠れ状態を維持してリアルタイム推論可能

#### 3. TinyLidarNetTCN (Temporal Convolutional Network)
```
[CNN Encoder] → [TCN Blocks (d=1,2,4)] → [FC Head]
                      ↓
             Dilated Causal Conv
             + Residual Connection
```

**TCNの利点**:
- 並列計算可能（LSTMより高速）
- Dilated Convolutionで長い依存関係をキャプチャ
- 学習が安定（勾配消失しにくい）

### データローダーの拡張

#### seq_len パラメータ
- `seq_len=1`: 従来の単一フレームモード（デフォルト）
- `seq_len>1`: 連続フレームのシーケンスを返すモード

#### 出力形式の変更
```
単一フレーム (seq_len=1):
  scan: (scan_dim,)
  odom: (13,)
  target: (2,)

シーケンス (seq_len>1):
  scans: (seq_len, scan_dim)
  odoms: (seq_len, 13)
  target: (2,)  ← 最後のフレームのターゲット
```

### 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `lib/data.py` | `seq_len`パラメータ追加、シーケンス返却モード |
| `lib/model.py` | `TinyLidarNetStacked`, `TinyLidarNetBiLSTM`, `TinyLidarNetTCN` 追加 |
| `train.py` | シーケンス学習対応、各モデル対応 |
| `model/tinylidarnet.py` | NumPy推論モデル追加（`*Np`クラス） |
| `tiny_lidar_net_controller_node.py` | フレームバッファ、LSTM状態管理 |
| `tiny_lidar_net_controller_core.py` | `process_sequence()`メソッド追加 |
| `convert_weight.py` | `--seq-len`, `--hidden-size` オプション追加 |
| `config/train.yaml` | 時系列モデル用パラメータ追加 |
| `config/param.yaml` | `seq_len`, `hidden_size` パラメータ追加 |

### 使用方法

#### 学習
```bash
# Frame Stacking
python train.py model.name='TinyLidarNetStacked' model.seq_len=10

# BiLSTM（学習時は未来情報も活用）
python train.py model.name='TinyLidarNetBiLSTM' model.hidden_size=128

# TCN（学習時はnon-causal）
python train.py model.name='TinyLidarNetTCN' model.tcn_causal=false
```

#### 重み変換
```bash
python convert_weight.py \
  --model tinylidarnet_bilstm \
  --seq-len 10 \
  --hidden-size 128 \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/bilstm_weights.npy
```

#### 推論設定 (param.yaml)
```yaml
model:
  architecture: "bilstm"  # "stacked", "bilstm", "tcn"
  seq_len: 10
  hidden_size: 128
```

### 推論時のバッファ管理

#### Frame Stacking / TCN
```python
# Node側でdequeでフレームバッファを管理
self._scan_buffer = deque(maxlen=seq_len)
self._odom_buffer = deque(maxlen=seq_len)

# バッファが満たされるまで待機
if len(self._scan_buffer) < self.seq_len:
    return  # Not enough frames yet
```

#### BiLSTM
```python
# LSTM隠れ状態を維持してリアルタイム推論
self.h, self.c = self._lstm_step(features, self.h, self.c)
# フレームごとに状態を更新しながら予測
```

### 比較表

| 項目 | Frame Stacking | BiLSTM | TCN |
|------|---------------|--------|-----|
| 実装難易度 | 低 | 中 | 中 |
| 推論速度 | 高速 | 中 | 高速 |
| 長期依存 | 低 | 高 | 中〜高 |
| 未来情報活用 | ✗ | ✓（学習時のみ） | ✓（non-causal時） |
| バッファ必要 | ✓ | ✗（状態維持） | ✓ |

---

## 📝 コミット履歴

| Hash | メッセージ |
|------|----------|
| `0eb182c` | feat(tiny_lidar_net): add mirror augmentation and fusion model support |

---

## 📚 参考

- [TinyLidarNet Paper (arXiv:2410.07447)](https://arxiv.org/abs/2410.07447)
- [CSL-KU/TinyLidarNet GitHub](https://github.com/CSL-KU/TinyLidarNet)
