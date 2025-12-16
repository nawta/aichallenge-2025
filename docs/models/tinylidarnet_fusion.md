# TinyLidarNetFusion

LiDARデータとkinematic state（odometry）を融合するマルチモーダルモデル。

## アーキテクチャ

```
入力:
  lidar: (batch, 1, 1080) - LiDARスキャン
  state: (batch, 13) - kinematic state

[LiDAR Branch - Same as TinyLidarNet]
  Conv1d(1→24, k=10, s=4) + ReLU   → 268
  Conv1d(24→36, k=8, s=4) + ReLU   → 66
  Conv1d(36→48, k=4, s=2) + ReLU   → 32
  Conv1d(48→64, k=3, s=1) + ReLU   → 30
  Conv1d(64→64, k=3, s=1) + ReLU   → 28
  Flatten → 1792

[State Branch]
  FC(13→64) + ReLU → 64

[Late Fusion]
  Concat(1792, 64) → 1856

[Regression Head]
  FC(1856→100) + ReLU
  FC(100→50) + ReLU
  FC(50→10) + ReLU
  FC(10→2) + Tanh

出力: (batch, 2) - [acceleration, steering]
```

## Kinematic State の構成

13次元のkinematic state ベクトル:

| インデックス | 名前 | 説明 | 正規化 |
|------------|------|------|--------|
| 0-2 | position | x, y, z 座標 | /100.0 |
| 3-6 | orientation | クォータニオン (x, y, z, w) | unchanged |
| 7-9 | linear_vel | 線速度 (vx, vy, vz) | /30.0 |
| 10-12 | angular_vel | 角速度 (wx, wy, wz) | /π |

## Late Fusion の設計

```
LiDAR Branch ────────┐
                     │
                     ├──→ Concat ──→ FC Head ──→ Output
                     │
State Branch ────────┘
```

**Late Fusion の利点**:
- 各モダリティが独立して特徴を学習
- 融合前に十分な抽象化
- 実装がシンプル

## 使用方法

### データ抽出

rosbagからodometryデータを抽出:

```bash
python extract_data_from_bag.py \
  --bag-path ./rosbag \
  --output-dir ./dataset \
  --odom-topic /localization/kinematic_state
```

### 学習

```bash
python train.py model.name='TinyLidarNetFusion'
```

### 重み変換

```bash
python convert_weight.py \
  --model tinylidarnet_fusion \
  --state-dim 13 \
  --ckpt ./checkpoints/best_model.pth \
  --output ./weights/fusion_weights.npy
```

### 推論設定

```yaml
model:
  architecture: "fusion"
  state_dim: 13
  ckpt_path: "/path/to/fusion_weights.npy"
```

## ROS 2 トピック

推論時に必要なトピック:

| トピック | 型 | 説明 |
|---------|---|------|
| `/scan` | sensor_msgs/LaserScan | LiDARスキャン |
| `/localization/kinematic_state` | nav_msgs/Odometry | 車両状態 |

## 特徴

- ✅ 車両状態を活用した予測
- ✅ 速度に応じた制御が可能
- ✅ LiDAR以外の情報を統合
- ❌ odometryの遅延に敏感
- ❌ ローカライゼーションの精度に依存

## 推奨ユースケース

- 速度情報が重要な制御
- 車両状態に応じた動的な応答
- ローカライゼーションが信頼できる環境
