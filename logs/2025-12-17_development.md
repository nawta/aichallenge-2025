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

## 🌙 一括学習スクリプト

### 背景
全モデル（7種類）をaugmented版・non-augmented版の両方で学習すると14回の学習が必要。
寝ている間に全モデルを学習し、朝起きたら全checkpointが揃っている状態にしたい。

### 作成したスクリプト

| ファイル | 説明 |
|---------|------|
| `train_all_models.sh` | Bash版（シンプル） |
| `train_all_models.py` | Python版（オプション豊富） |

### 学習されるモデル（14種類）

| モデル | Augmented | Non-Augmented |
|--------|-----------|---------------|
| TinyLidarNet | ✅ | ✅ |
| TinyLidarNetSmall | ✅ | ✅ |
| TinyLidarNetDeep | ✅ | ✅ |
| TinyLidarNetFusion | ✅ | ✅ |
| TinyLidarNetStacked | ✅ | ✅ |
| TinyLidarNetBiLSTM | ✅ | ✅ |
| TinyLidarNetTCN | ✅ | ✅ |

### 使用方法

```bash
# Dockerコンテナ内で実行
cd /aichallenge/python_workspace/tiny_lidar_net

# GPU使用（推奨）
./train_all_models.sh

# CPUのみ（RTX 50シリーズ等）
./train_all_models.sh --cpu

# Python版（より柔軟）
python3 train_all_models.py --epochs 50 --models TinyLidarNet TinyLidarNetDeep
```

### 出力構造

```
checkpoints/
├── TinyLidarNet_aug/
│   ├── best_model.pth
│   └── last_model.pth
├── TinyLidarNet_noaug/
├── TinyLidarNetDeep_aug/
└── ...

weights/
├── TinyLidarNet_aug.npy      # 推論用（変換済み）
├── TinyLidarNet_noaug.npy
└── ...

training_logs/
├── TinyLidarNet_aug_20251217_xxxx.log
├── summary_20251217_xxxx.txt  # 全体サマリー
└── ...
```

### 結果確認

```bash
# 朝起きたら
cat /aichallenge/python_workspace/tiny_lidar_net/training_logs/summary_*.txt

# 変換済み重みの確認
ls -la /aichallenge/python_workspace/tiny_lidar_net/weights/*.npy
```

### スクリプトの特徴

- ✅ 学習後に自動で重み変換（.pth → .npy）
- ✅ 各モデルの学習ログを個別ファイルに保存
- ✅ 全体のサマリーを生成
- ✅ エラー時も次のモデルの学習を継続
- ✅ CPU/GPU切り替え対応

---

## 📝 コミット履歴

| Hash | メッセージ |
|------|----------|
| `5f8178d` | feat(tiny_lidar_net): add overnight training script for all models |
| `63a1b57` | feat(tiny_lidar_net): add temporal models (Stacked, BiLSTM, TCN) |
| `0eb182c` | feat(tiny_lidar_net): add mirror augmentation and fusion model support |

---

---

## 🗺️ BEV Map Encoder Ablation Study

### 背景
事前スキャンされたマップ情報（`lane.csv`）をTinyLiDARNetに統合し、走行性能を向上させる。
Ablation Study用に3つのBEVエンコーダパターンを実装。

### BEV (Bird's Eye View) とは
車両周辺の環境を鳥瞰図として表現した2Dグリッド。車線境界をラスタライズして入力特徴量として使用。

### 3つのパターン

| Pattern | Architecture | BEV Type | Grid Size | Channels | 特徴 |
|---------|--------------|----------|-----------|----------|------|
| **A** | `local_bev` | Local | 64×64 | 2 | 車両中心、yaw回転あり |
| **B** | `global_bev` | Global | 128×128 | 3 | マップ固定座標、回転なし |
| **C** | `dual_bev` | Both | 64×64 + 128×128 | 2 + 3 | 両方を統合 |

### Local BEV vs Global BEV

#### Local BEV（パターンA）
```
特徴:
- 車両位置を中心とした局所座標系
- 車両のyaw角に合わせて回転（前方が常に上）
- 近傍の車線境界を高解像度でキャプチャ
- Channel 0: 左境界, Channel 1: 右境界

用途:
- 局所的な障害物回避
- レーン追従
- 直近の道路形状把握
```

#### Global BEV（パターンB）
```
特徴:
- マップ固定座標系（回転なし）
- より広い範囲をカバー（192m × 192m）
- 自車位置を3チャンネル目にマーカーとして描画
- Channel 0: 左境界, Channel 1: 右境界, Channel 2: 自車位置

用途:
- 大局的な経路計画
- コース全体の把握
- 先の曲がり角の認識
```

### アーキテクチャ詳細

#### Pattern A: TinyLidarNetLocalBEV
```
入力:
  - lidar: (batch, 1, input_dim)
  - local_bev: (batch, 2, 64, 64)
  - state: (batch, 13)

[LiDAR Branch]
  Conv1D(1→24→36→48→64→64) → Flatten → 1792

[Local BEV Branch]
  Conv2D(2→16→32→64) stride=2, padding=1
  → Flatten → FC(→256)

[State Branch]
  FC(13→64)

[Fusion]
  Concat(1792 + 256 + 64) → FC Head → Output (2)
```

#### Pattern B: TinyLidarNetGlobalBEV
```
入力:
  - lidar: (batch, 1, input_dim)
  - global_bev: (batch, 3, 128, 128)
  - state: (batch, 13)

[LiDAR Branch]
  (same as above)

[Global BEV Branch]
  Conv2D(3→16→32→64→64) stride=2, padding=1  # 4 layers for 128x128
  → Flatten → FC(→256)

[State Branch]
  FC(13→64)

[Fusion]
  Concat(1792 + 256 + 64) → FC Head → Output (2)
```

#### Pattern C: TinyLidarNetDualBEV
```
入力:
  - lidar: (batch, 1, input_dim)
  - local_bev: (batch, 2, 64, 64)
  - global_bev: (batch, 3, 128, 128)
  - state: (batch, 13)

[LiDAR Branch]
  → 1792

[Local BEV Branch]
  → 256

[Global BEV Branch]
  → 256

[State Branch]
  → 64

[Fusion]
  Concat(1792 + 256 + 256 + 64 = 2368)
  → FC(256) → FC(64) → FC(10) → FC(2)
```

### 座標変換

#### 問題
`lane.csv`の座標とLocalization座標（`/localization/kinematic_state`）は両方ともMGRS座標系。
値が大きすぎるため、共通のオフセットで正規化が必要。

#### 解決策（laserscan_generatorと同じ方式）
```cpp
// lane.csvの最初の点をオフセットとして使用
if (!is_offset_initialized_) {
    map_offset_ = first_point;
    is_offset_initialized_ = true;
}

// マップ座標に適用
map_point.x -= map_offset_.x;
map_point.y -= map_offset_.y;

// Localization座標にも同じオフセットを適用
ego_x -= map_offset_.x;
ego_y -= map_offset_.y;
```

### 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `bev_generator.py` | `generate_local()`, `generate_global()` メソッド追加 |
| `model/tinylidarnet.py` | 6クラス追加（PyTorch 3 + NumPy 3） |
| `tiny_lidar_net_controller_core.py` | BEVアーキテクチャ対応、`process_with_bev()` 等 |
| `tiny_lidar_net_controller_node.py` | マップ読み込み、BEV生成、座標変換統合 |
| `config/tiny_lidar_net_node.param.yaml` | BEVパラメータ追加 |

### 使用方法

#### パラメータ設定
```yaml
# tiny_lidar_net_node.param.yaml
model:
  architecture: "local_bev"  # or "global_bev" or "dual_bev"
  ckpt_path: "/path/to/trained_weights.npy"

bev:
  map_path: "/path/to/lane.csv"  # 必須
  local_size: 64
  local_resolution: 1.0      # 64m × 64m カバー
  local_channels: 2
  global_size: 128
  global_resolution: 1.5     # 192m × 192m カバー
  global_channels: 3
```

#### lane.csv の生成
```bash
# osm2csv.py を使用（既存ツール）
python3 osm2csv.py \
  /path/to/lanelet2_map.osm \
  /path/to/lane.csv
```

### Ablation Study 計画

| 実験 | Architecture | 仮説 |
|------|--------------|------|
| Baseline | `large` (LiDAR only) | ベースライン |
| Exp A | `local_bev` | 局所的な車線追従が向上 |
| Exp B | `global_bev` | 先読みによりコーナリング改善 |
| Exp C | `dual_bev` | 両方の利点を統合 |

### 期待される効果

1. **レーン逸脱防止**: 車線境界を明示的に入力することで、逸脱を減らす
2. **コーナリング改善**: 先のカーブを事前に認識し、早めの減速・ステアリング
3. **ドメインギャップ軽減**: マップ情報は学習/推論環境で共通

---

## 🗺️ TinyLidarNetMap - 静的マップ画像統合

### 背景
`map_image/2.png`（トラック境界線・走行経路付き処理済み画像）を2D CNNエンコーダーで特徴抽出し、LiDARブランチとLate Fusionで統合する新モデル`TinyLidarNetMap`を実装。

BEVモデル（`local_bev`, `global_bev`, `dual_bev`）はリアルタイムでBEVを生成するのに対し、このモデルは**静的なマップ画像を起動時に1回だけエンコードしてキャッシュ**することで、推論時のオーバーヘッドを最小化。

### アーキテクチャ

```
入力:
  - LiDAR: (batch, 1, 1080)
  - Map Image: (1, 3, 224, 224)  ← 起動時に1回だけエンコード

[LiDAR Branch]
  Conv1D(1→24→36→48→64→64) → Flatten → 1792

[Map Branch - MapEncoder]
  Conv2D(3→32, k=7, s=2, p=3) + BN + ReLU → MaxPool(2×2)  → 56×56
  Conv2D(32→64, k=5, s=2, p=2) + BN + ReLU → MaxPool(2×2) → 14×14
  Conv2D(64→128, k=3, s=1, p=1) + BN + ReLU → MaxPool(2×2) → 7×7
  Conv2D(128→128, k=3, s=1, p=1) + BN + ReLU
  → Global Average Pool → FC(128→128) → ReLU
  → 128-dim features (cached)

[Late Fusion]
  Concat(1792 + 128 = 1920) → FC Head → Output (2)
```

### 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `lib/model.py` | `MapEncoder`, `TinyLidarNetMap` クラス追加 |
| `train.py` | `TinyLidarNetMap` 対応、`load_map_image()` 追加 |
| `convert_weight.py` | `tinylidarnet_map` オプション追加 |
| `model/numpy/layers.py` | `batch_norm2d`, `conv2d_padded`, `adaptive_avg_pool2d` 追加 |
| `model/__init__.py` | 新関数のエクスポート追加 |
| `model/tinylidarnet.py` | `MapEncoderImage`, `TinyLidarNetMapImage`, `TinyLidarNetMapImageNp` 追加 |
| `tiny_lidar_net_controller_core.py` | `map_image` アーキテクチャ対応 |
| `tiny_lidar_net_controller_node.py` | `map.image_path`, `map.feature_dim` パラメータ追加 |

### 使用方法

#### 学習
```yaml
# config/train.yaml
model:
  name: TinyLidarNetMap
  map_image_path: "../../map_image/2.png"
  map_feature_dim: 128
```

```bash
python train.py model.name='TinyLidarNetMap' model.map_image_path='../../map_image/2.png'
```

#### 重み変換
```bash
python convert_weight.py \
  --model tinylidarnet_map \
  --map-feature-dim 128 \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/tinylidarnet_map.npy
```

#### 推論設定
```yaml
# tiny_lidar_net_node.param.yaml
model:
  architecture: "map_image"
  ckpt_path: "/path/to/tinylidarnet_map.npy"

map:
  image_path: "/path/to/map_image/2.png"
  feature_dim: 128
```

### 設計ポイント

1. **静的キャッシュ**: マップは変化しないため、起動時に1回だけエンコードしてキャッシュ。推論時は毎回同じ特徴量を再利用。
2. **Late Fusion**: LiDAR特徴量（1792次元）とマップ特徴量（128次元）を連結してFC層に入力。
3. **標準サイズ**: 224×224にリサイズ（CNN設計の標準サイズ、計算効率）
4. **BatchNorm**: 2D BatchNormを使用して学習の安定化

### BEVモデルとの比較

| 項目 | TinyLidarNetMap | BEV Models |
|------|-----------------|------------|
| マップ表現 | RGB画像 (224×224) | BEVグリッド (64×64/128×128) |
| 更新頻度 | 起動時1回 | 毎フレーム |
| 座標系 | なし（画像全体） | 車両中心/マップ固定 |
| 計算コスト | 低（キャッシュ） | 中〜高 |
| 自車位置反映 | なし | あり |

### 期待される効果

1. **グローバルコンテキスト**: トラック全体の形状を把握
2. **軽量推論**: マップ特徴量は事前計算済みでメモリから読み出すだけ
3. **シンプルな実装**: リアルタイム座標変換が不要

---

## 🔧 train_all_models.sh 修正 & BEV学習パイプライン統合

### 問題分析

#### 発見した問題
`train_all_models.sh` を実行したところ、時系列モデル（TinyLidarNetStacked, BiLSTM, TCN）が全て失敗していた。

```
Single-frame models: 成功 (1200-2700秒)
Temporal models: 失敗 (1-2秒で終了、"No best_model.pth found")
```

#### 根本原因
**エラー:** `train.py: error: unrecognized arguments: --seq-len 10 --hidden-size 128`

スクリプトがHydra形式（ドット記法）とCLIフラグを混在させていた：
```bash
# 間違い - 形式が混在
EXTRA_ARGS="model.seq_len=10 --seq-len 10"
```

`train.py` はHydraを使用しており、ドット記法のオーバーライドのみ受け付ける。

### 実施した修正

#### Commit 1: Priority 1 & 2 修正 (2408bf5)

##### 1. config/train.yaml
- `input_dim: 750` → `input_dim: 1080` に変更（モデルデフォルトに合わせる）

##### 2. convert_weight.py
- TCNモデル推論用に `--tcn-causal` 引数を追加

##### 3. train_all_models.sh
- **重要な修正:** train引数とconvert引数を分離：
  ```bash
  train_model() {
      local TRAIN_EXTRA_ARGS=$4    # train.py用Hydraオーバーライド
      local CONVERT_EXTRA_ARGS=$5  # convert_weight.py用CLI引数
  }
  ```
- 時系列モデル用：
  ```bash
  TRAIN_EXTRA="model.seq_len=${SEQ_LEN} model.hidden_size=${HIDDEN_SIZE}"
  CONVERT_EXTRA="--seq-len ${SEQ_LEN} --hidden-size ${HIDDEN_SIZE}"
  ```

##### 4. TinyLidarNetMap学習追加
- 両方のマップ画像（1.png, 2.png）で学習
- 各マップ用に別のチェックポイントディレクトリ

---

#### Commit 2: BEV学習パイプライン実装 (f6a821e)

##### 新規作成ファイル

**lib/bev_generator.py**
- `BEVGenerator` クラス: `generate_local()`, `generate_global()`, `generate_both()` メソッド
- Local BEV: 64×64グリッド、2チャンネル（左/右境界）、車両中心、yaw回転あり
- Global BEV: 128×128グリッド、3チャンネル（左/右/自車マーカー）、マップ固定座標
- Bresenhamのラインアルゴリズムで効率的なグリッド描画
- `quaternion_to_yaw()` ユーティリティ関数

**lib/map_loader.py**
- `LaneBoundaries` データクラス: 境界データの整理された格納
- `load_lane_boundaries()`: CSVパース関数
- `get_nearby_boundaries()`: 空間クエリ用
- 座標オフセットの自動/手動正規化

##### 変更ファイル

**lib/model.py** (+410行)
- `TinyLidarNetLocalBEV`: LiDAR (5 Conv1D) + Local BEV (3 Conv2D) + State (FC) → Late Fusion
- `TinyLidarNetGlobalBEV`: LiDAR + Global BEV (4 Conv2D) + State → Late Fusion
- `TinyLidarNetDualBEV`: LiDAR + 両BEV + State → Late Fusion（最大モデル）
- 全てMLPヘッド前の連結によるLate Fusionアーキテクチャ

**lib/data.py** (+336行)
- `BEVScanControlSequenceDataset`: BEV生成付き単一シーケンスデータセット
- `BEVMultiSeqConcatDataset`: BEV用マルチシーケンス連結
- `bev_mode` サポート: 'local', 'global', 'both'
- ミラー拡張: スキャン・BEV水平反転、左右チャンネル入れ替え、ステアリング符号反転

**train.py** (+147行)
- `BEV_MODELS` 定数追加
- モデル名からBEVモード自動検出
- `BEVMultiSeqConcatDataset` でのBEV専用データセット作成
- 学習/検証ループでのBEV専用バッチ展開

**convert_weight.py** (+58行)
- argparseにBEVモデル選択肢追加
- `--local-grid-size` と `--global-grid-size` 引数追加
- `load_model()` でのBEVモデルインスタンス化

**train_all_models.sh** (+46行)
- 3モデルの `BEV_MODELS` 配列追加
- BEVパラメータ（グリッドサイズ、解像度、lane CSVパス）
- BEV学習セクション追加

### 学習可能なモデル一覧

| カテゴリ | モデル | aug/noaug |
|----------|--------|-----------|
| Single-frame | TinyLidarNet, Small, Deep, Fusion | 4 × 2 = 8 |
| Temporal | Stacked, BiLSTM, TCN | 3 × 2 = 6 |
| Map | TinyLidarNetMap (×2 maps) | 1 × 2 × 2 = 4 |
| BEV | LocalBEV, GlobalBEV, DualBEV | 3 × 2 = 6 |
| **合計** | | **24回** |

### 重要なパス

- Lane CSV: `/aichallenge/workspace/src/aichallenge_submit/laserscan_generator/map/lane.csv`
- Map images: `/aichallenge/map_image/1.png`, `/aichallenge/map_image/2.png`
- Checkpoints: `checkpoints/{ModelName}_{aug|noaug}/`
- Weights: `weights/{ModelName}_{aug|noaug}.npy`

### 使用方法

#### 時系列モデル修正テスト
```bash
# "unrecognized arguments" エラーなく動作するはず
python3 train.py model.name='TinyLidarNetStacked' model.seq_len=10 model.hidden_size=128
```

#### BEVモデルテスト
```bash
# 単一BEVモデル
python3 train.py \
  model.name='TinyLidarNetLocalBEV' \
  model.lane_csv_path='/aichallenge/workspace/src/aichallenge_submit/laserscan_generator/map/lane.csv'

# 全モデル一括学習（一晩）
./train_all_models.sh
```

### Gitコミット

| Hash | メッセージ |
|------|------------|
| `2408bf5` | fix(tiny_lidar_net): separate Hydra/CLI args and add TinyLidarNetMap training |
| `f6a821e` | feat(tiny_lidar_net): add BEV model support for training |
| `27c632d` | docs: add development log for 2025/12/17 |
| `2fe17c4` | test(tiny_lidar_net): add quick test script for all models |

### クイックテストスクリプト

全モデルの動作確認用スクリプト `test_all_models.sh` を作成:

```bash
# Dockerコンテナ内で実行
cd /aichallenge/python_workspace/tiny_lidar_net
./test_all_models.sh
```

- 1エポック、小バッチサイズでテスト
- 全11モデル（Single-frame 4 + Temporal 3 + Map 1 + BEV 3）をテスト
- 成功/失敗のサマリーを表示

---

## 📚 参考

- [TinyLidarNet Paper (arXiv:2410.07447)](https://arxiv.org/abs/2410.07447)
- [CSL-KU/TinyLidarNet GitHub](https://github.com/CSL-KU/TinyLidarNet)
