---
marp: true
theme: default
paginate: true
header: 'AI Challenge 2025 - TinyLiDARNet Task'
---

<style>
blockquote {
  position: absolute !important;
  bottom: 20px !important;
  left: 40px !important;
  font-size: 0.4em !important;
  border-left: 3px solid #ccc !important;
  text-align: left !important;
  background: transparent !important;
  padding-left: 10px !important;
  margin: 0 !important;
  color: #666 !important;
  width: auto !important;
  max-width: 40% !important;
}
blockquote::before {
  content: none !important;
  display: none !important;
}
blockquote::after {
  content: none !important;
  display: none !important;
}
blockquote p {
  margin: 0 !important;
  padding: 0 !important;
  line-height: 1.4 !important;
}
</style>

<!-- _class: lead -->
# TinyLiDARNet MLOps チャレンジ

---

## 目次

1. タスクの概要
2. TinyLiDARNetとは
3. 実装ステップ
4. 発展的課題
5. まとめ

---

<!-- _class: lead -->
# 1. タスクの概要

---

## タスクの目標

### 🏁 ML Modelを学習させ、参加者内で最も高速な周回を実現すること

- **評価指標**: 周回タイム

### 得られるスキル・経験

- End-to-Endな自動運転の実践的な学習
- データ収集から学習、デプロイまでのMLOpsにおける一連の流れを体験
- 効率的・高精度なデータ学習のための既存アルゴリズム(e.g. Pure Pursuit)の学習

---

<!-- _class: lead -->
# 2. TinyLiDARNetとは

---

## TinyLiDARNet: 概要

- 軽量なEnd-to-Endモデル: 2D LiDARデータから速度とステアリング角を直接推定

### 入力・出力

- **入力**: 長さ1081の1次元配列（2D LiDARの距離データ）
- **出力**: 速度 (Speed) とステアリング角 (Steering Angle)
    - defaultでは、速度は固定値で制御

### 特徴

- **シンプル・軽量**: 5層のCNN + 4層の全結合層
- **性能**: 学習すればサーキットを走行可能

---

## TinyLiDARNet: Architecture

<div style="display: flex; align-items: center;">
<div style="flex: 1;">

**特徴**
- 5層のCNN
- 4層の全結合層
- シンプルで軽量

**論文**: [TinyLidarNet](https://arxiv.org/abs/2410.07447)

![h:600 bg right](https://github.com/CSL-KU/TinyLidarNet/raw/main/Images/TinyLidarNet_Architecture.jpg)

> 図の引用: https://arxiv.org/abs/2410.07447
---

<!-- _class: lead -->
# 3. MLOpsの実践ステップ

---

## MLOps ワークフロー

**You design it**
- Modelの設計: TinyLiDARNet

**You train it**
- データ収集・変換
- 学習(Training & Validation)

**You run it**
- Deploy・推論(Autowareで実行)
- 発見した課題を、Model設計やデータ変換へfeedback

![h:300 bg right](https://github.com/visenger/awesome-mlops/raw/master/awesome-mlops-intro.png)

> 図の引用: https://github.com/visenger/awesome-mlops

---

## Step 1: Rule Basedでデータ収集

### データ収集の流れ

- **AWSIM**(シミュレータ)内で、**Autoware**を用いてRule-basedな自動走行を実行
- センサデータと制御コマンドを**ROSBAG**として記録

### 必要なデータ

- 2D LiDARのスキャンデータ (`/sensing/lidar/scan_generator/concatenated_cloud`)
- ステアリング角 (`/control/command/control_cmd`)

---

## Step 1: データ収集の実行例

### 方法1: 手動運転での収集

```bash
# Terminal 1: Scan generation node
ros2 launch laserscan_generator laserscan_generator.launch.xml \
  use_sim_time:=true \
  csv_path:=$(ros2 pkg prefix laserscan_generator)/share/laserscan_generator/map/lane.csv

# Terminal 2: AWSIM起動
./run_simulator.bash

# Terminal 3: joycon接続
ros2 launch teleop_manager teleop_manager.launch.xml

# Terminal 4: ROSbag記録
./record_rosbag.bash
```

**Tips**: 複数のシナリオ（アウトインアウト、インアウトインでのコーナリング）で収集

---

## Step 1: データ収集の実行例

### 方法2: autowareでの収集

```bash
# Terminal 1: Scan generation node
ros2 launch laserscan_generator laserscan_generator.launch.xml \
  use_sim_time:=true \
  csv_path:=$(ros2 pkg prefix laserscan_generator)/share/laserscan_generator/map/lane.csv

# Terminal 2: AWSIM起動
./run_simulator.bash

# Terminal 3: Autoware起動
./run_autoware.bash awsim

# Terminal 4: ROSbag記録
./record_rosbag.bash
```

**Tips**: autoware(Pure Pursuit)をうまくtuningして、安定した走行を実現する必要あり。[入門講座](https://automotiveaichallenge.github.io/aichallenge-documentation-2025/course/velocity_planning.html#02-04)を参考にtuningしてください。

---

## Step 2: データ変換

### ROSbagから学習用データセットへ変換

```bash
python3 extract_data_from_bag.py \
  --bags-dir /aichallenge/rosbag2_autoware_train_01/ \
  --outdir /aichallenge/python_workspace/tiny_lidar_net/dataset/train/
```

### データセットの構成

- **Train set**: モデルの学習用
- **Validation set**: ハイパーパラメータ調整・過学習防止用

**Tips**: leakが起こらないように分割する

---

## Step 3: モデルの学習

### 学習の実行

```bash
# GPU使用時
python3 /aichallenge/python_workspace/tiny_lidar_net/train.py

# CPU使用時
CUDA_VISIBLE_DEVICES="" python3 /aichallenge/python_workspace/tiny_lidar_net/train.py
```

### 学習のポイント

- **Epoch数**: 過学習に注意しながら調整
- **Batch size**: メモリに応じて調整
- **Learning rate**: 学習の安定性を確認

---

## Step 4: モデルのデプロイ

### 1. モデル形式の変換（.pth → .npy）

```bash
python3 convert_weight.py \
  --ckpt /aichallenge/python_workspace/tiny_lidar_net/checkpoints/best_model.pth \
  --output /aichallenge/python_workspace/tiny_lidar_net/weights/converted_weights.npy
```

### 2. ROS 2 Packageへの配置

```bash
mv /aichallenge/python_workspace/tiny_lidar_net/weights/converted_weights.npy \
   /aichallenge/workspace/src/aichallenge_submit/tiny_lidar_net_controller/ckpt/tinylidarnet_weights.npy
```

---

## Step 4: 実行

### Control modeの変更

`reference.launch.xml`のcontrol modeを`rule_based`から`e2e`に変更

### TinyLiDARNetでの走行

```bash
# Terminal 1: AWSIM起動
./run_simulator.bash

# Terminal 2: Autoware起動（E2E mode）
./run_autoware.bash awsim
```

### 評価項目

- 安定して周回できるか(コースアウトやクラッシュはないか)
- 周回タイムはどの程度か

---

<!-- _class: lead -->
# 4. 発展的課題

---

## 発展的課題 1: Pitlane走行

### 課題

- TinyLiDARNetは広いコース上では走行可能
- しかし、Pitlaneのような**狭いエリアでの走行は苦手**

### アプローチ

1. [Pitlane用AWSIM](https://tier4inc-my.sharepoint.com/...)を使用
2. Pitlane走行のROSBAGを収集
3. TinyLiDARNetを学習

![h:600 bg right](../assets/tiny_lidar_net_awsim_pitlane.png)

---

## 発展的課題 2: アクセル制御の追加

### 現状の制限

- デフォルトではステアリング制御のみ
- アクセルは固定値で制御

### 改善案

- `control_mode: "ai"`に変更
- アクセル制御もTinyLiDARNetで実施
- より柔軟な速度調整が可能に

### 期待される効果

- カーブでの減速、直線での加速の最適化
- さらなる周回タイムの短縮

---

## 発展的課題 3: Overtake

### 課題

- 複数台走行では高度な意思決定が必要
- 戦略的な行動(Overtake)が必須

### アプローチ

1. [複数台走行用AWSIM](https://tier4inc-my.sharepoint.com/...)を使用
2. 他車両との相互作用を含むROSBAGを収集
3. TinyLiDARNetを学習

![h:270 bg right](../assets/tiny_lidar_net_awsim_multi_car.png)

---

## 発展的課題 4: 画像input

### TinyLiDARImageNetへの拡張

- 現状: LiDARデータのみを使用
- 拡張案: カメラ画像も入力に追加

### メリット

- 信号や標識の認識が可能に
- 黄旗や手信号などへの対応

![h:270 bg right](../assets/camera_awsim_after.png)

---

<!-- _class: lead -->
# 5. まとめ

---

## まとめ

### TinyLiDARNetタスクで学べること

1. **End-to-End自動運転**の実践的な経験
2. **MLOps**: データ収集 → データ変換 → 学習 → デプロイ

---

<!-- _class: lead -->
# Thank you!

## Questions?

参考資料:
- [Getting Started: TinyLiDARNet](../ml_sample/getting_started_tiny_lidar_net.md)
- [Algorithms](../ml_sample/algorithms.md)
- [論文: TinyLidarNet](https://arxiv.org/abs/2410.07447)
