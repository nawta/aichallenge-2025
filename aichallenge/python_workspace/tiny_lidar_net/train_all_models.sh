#!/bin/bash
# =============================================================================
# TinyLidarNet - Train All Models Script
# =============================================================================
# This script trains all model architectures with and without data augmentation.
# Run this overnight to get all checkpoints ready in the morning.
#
# Usage:
#   ./train_all_models.sh           # Use GPU
#   ./train_all_models.sh --cpu     # Use CPU only
#
# Output:
#   checkpoints/
#   ├── TinyLidarNet_aug/
#   ├── TinyLidarNet_noaug/
#   ├── TinyLidarNetSmall_aug/
#   ├── ...
#   weights/
#   ├── TinyLidarNet_aug.npy
#   ├── TinyLidarNet_noaug.npy
#   ├── ...
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_BASE="${SCRIPT_DIR}/checkpoints"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
LOG_DIR="${SCRIPT_DIR}/training_logs"

# Default input dimension (must match config/train.yaml)
INPUT_DIM=1080

# Training epochs (override default from config)
EPOCHS=200

# Check for CPU-only mode
USE_CPU=""
if [[ "$1" == "--cpu" ]]; then
    USE_CPU="CUDA_VISIBLE_DEVICES=\"\""
    echo "🖥️  Running in CPU-only mode"
fi

# Create directories
mkdir -p "${CHECKPOINT_BASE}"
mkdir -p "${WEIGHTS_DIR}"
mkdir -p "${LOG_DIR}"

# Timestamp for this training session
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="${LOG_DIR}/training_summary_${TIMESTAMP}.txt"

echo "=============================================" | tee -a "${SUMMARY_FILE}"
echo "TinyLidarNet Training Session" | tee -a "${SUMMARY_FILE}"
echo "Started: $(date)" | tee -a "${SUMMARY_FILE}"
echo "=============================================" | tee -a "${SUMMARY_FILE}"

# =============================================================================
# Model Definitions
# =============================================================================

# Single-frame models
SINGLE_FRAME_MODELS=(
    "TinyLidarNet:tinylidarnet"
    "TinyLidarNetSmall:tinylidarnet_small"
    "TinyLidarNetDeep:tinylidarnet_deep"
    "TinyLidarNetFusion:tinylidarnet_fusion"
)

# Temporal models (require seq_len)
TEMPORAL_MODELS=(
    "TinyLidarNetStacked:tinylidarnet_stacked"
    "TinyLidarNetBiLSTM:tinylidarnet_bilstm"
    "TinyLidarNetTCN:tinylidarnet_tcn"
)

# Map models
MAP_MODELS=(
    "TinyLidarNetMap:tinylidarnet_map"
)

# Map images to train with
MAP_IMAGES=(
    "/aichallenge/map_image/1.png"
    "/aichallenge/map_image/2.png"
)

# BEV models (require lane_csv_path)
BEV_MODELS=(
    "TinyLidarNetLocalBEV:tinylidarnet_local_bev"
    "TinyLidarNetGlobalBEV:tinylidarnet_global_bev"
    "TinyLidarNetDualBEV:tinylidarnet_dual_bev"
)

# Lane CSV path for BEV models
LANE_CSV_PATH="/aichallenge/workspace/src/aichallenge_submit/laserscan_generator/map/lane.csv"

# BEV parameters
LOCAL_BEV_SIZE=64
LOCAL_BEV_CHANNELS=2
LOCAL_RESOLUTION=1.0
GLOBAL_BEV_SIZE=128
GLOBAL_BEV_CHANNELS=3
GLOBAL_RESOLUTION=1.5

# =============================================================================
# Training Function
# =============================================================================
# Arguments:
#   $1 - MODEL_NAME: Name of the model (e.g., "TinyLidarNet")
#   $2 - CONVERT_NAME: Name for convert_weight.py (e.g., "tinylidarnet")
#   $3 - AUGMENT: "true" or "false"
#   $4 - TRAIN_EXTRA_ARGS: Extra Hydra overrides for train.py (e.g., "model.seq_len=10")
#   $5 - CONVERT_EXTRA_ARGS: Extra CLI args for convert_weight.py (e.g., "--seq-len 10")
#   $6 - SAVE_SUFFIX: Optional suffix for checkpoint directory (e.g., "_map1")

train_model() {
    local MODEL_NAME=$1
    local CONVERT_NAME=$2
    local AUGMENT=$3
    local TRAIN_EXTRA_ARGS=$4
    local CONVERT_EXTRA_ARGS=$5
    local SAVE_SUFFIX=${6:-""}

    local AUG_SUFFIX=""
    if [[ "$AUGMENT" == "true" ]]; then
        AUG_SUFFIX="_aug"
    else
        AUG_SUFFIX="_noaug"
    fi

    local FULL_NAME="${MODEL_NAME}${SAVE_SUFFIX}${AUG_SUFFIX}"
    local SAVE_DIR="${CHECKPOINT_BASE}/${FULL_NAME}"
    local LOG_FILE="${LOG_DIR}/${FULL_NAME}_${TIMESTAMP}.log"
    local WEIGHT_FILE="${WEIGHTS_DIR}/${FULL_NAME}.npy"

    echo "" | tee -a "${SUMMARY_FILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${SUMMARY_FILE}"
    echo "🚀 Training: ${FULL_NAME}" | tee -a "${SUMMARY_FILE}"
    echo "   Augmentation: ${AUGMENT}" | tee -a "${SUMMARY_FILE}"
    echo "   Save Dir: ${SAVE_DIR}" | tee -a "${SUMMARY_FILE}"
    echo "   Started: $(date)" | tee -a "${SUMMARY_FILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${SUMMARY_FILE}"

    # Build training command (Hydra overrides only)
    # Note: early_stop_patience=null disables early stopping (train all epochs)
    local CMD="${USE_CPU} python3 ${SCRIPT_DIR}/train.py \
        model.name='${MODEL_NAME}' \
        data.augment_mirror=${AUGMENT} \
        train.save_dir='${SAVE_DIR}' \
        train.epochs=${EPOCHS} \
        train.early_stop_patience=null \
        ${TRAIN_EXTRA_ARGS}"

    # Run training
    START_TIME=$(date +%s)

    if eval ${CMD} 2>&1 | tee "${LOG_FILE}"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "✅ Training completed in ${DURATION}s" | tee -a "${SUMMARY_FILE}"

        # Convert weights
        if [[ -f "${SAVE_DIR}/best_model.pth" ]]; then
            echo "📦 Converting weights..." | tee -a "${SUMMARY_FILE}"

            # Build convert command (CLI args only)
            local CONVERT_CMD="python3 ${SCRIPT_DIR}/convert_weight.py \
                --model ${CONVERT_NAME} \
                --input-dim ${INPUT_DIM} \
                --ckpt '${SAVE_DIR}/best_model.pth' \
                --output '${WEIGHT_FILE}' \
                ${CONVERT_EXTRA_ARGS}"

            if eval ${CONVERT_CMD}; then
                echo "✅ Weights saved: ${WEIGHT_FILE}" | tee -a "${SUMMARY_FILE}"
            else
                echo "❌ Weight conversion failed" | tee -a "${SUMMARY_FILE}"
            fi
        else
            echo "⚠️  No best_model.pth found" | tee -a "${SUMMARY_FILE}"
        fi
    else
        echo "❌ Training failed! Check log: ${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
    fi

    echo "   Finished: $(date)" | tee -a "${SUMMARY_FILE}"
}

# =============================================================================
# Main Training Loop
# =============================================================================

# Count total training runs
TOTAL_SINGLE=$((${#SINGLE_FRAME_MODELS[@]} * 2))
TOTAL_TEMPORAL=$((${#TEMPORAL_MODELS[@]} * 2))
TOTAL_MAP=$((${#MAP_MODELS[@]} * ${#MAP_IMAGES[@]} * 2))
TOTAL_BEV=$((${#BEV_MODELS[@]} * 2))
TOTAL_RUNS=$((TOTAL_SINGLE + TOTAL_TEMPORAL + TOTAL_MAP + TOTAL_BEV))

echo ""
echo "📋 Training Plan:"
echo "   Epochs per model: ${EPOCHS}"
echo "   1. Map models (PRIORITY): ${#MAP_MODELS[@]} × ${#MAP_IMAGES[@]} maps × 2 (aug/noaug) = ${TOTAL_MAP}"
echo "   2. Single-frame models: ${#SINGLE_FRAME_MODELS[@]} × 2 (aug/noaug) = ${TOTAL_SINGLE}"
echo "   3. Temporal models: ${#TEMPORAL_MODELS[@]} × 2 (aug/noaug) = ${TOTAL_TEMPORAL}"
echo "   4. BEV models: ${#BEV_MODELS[@]} × 2 (aug/noaug) = ${TOTAL_BEV}"
echo "   Total: ${TOTAL_RUNS} training runs"
echo ""

# =============================================================================
# Train Map Models (PRIORITY - train first)
# =============================================================================

echo "=========================================="
echo "🔹 Training Map Models (Priority)"
echo "=========================================="

MAP_FEATURE_DIM=128

for MAP_IMAGE in "${MAP_IMAGES[@]}"; do
    # Extract map name from path (e.g., "1" from "/aichallenge/map_image/1.png")
    MAP_NAME=$(basename "${MAP_IMAGE}" .png)

    for MODEL_INFO in "${MAP_MODELS[@]}"; do
        MODEL_NAME="${MODEL_INFO%%:*}"
        CONVERT_NAME="${MODEL_INFO##*:}"

        # Hydra overrides for train.py
        TRAIN_EXTRA="model.map_image_path='${MAP_IMAGE}' model.map_feature_dim=${MAP_FEATURE_DIM}"

        # CLI args for convert_weight.py
        CONVERT_EXTRA="--map-feature-dim ${MAP_FEATURE_DIM}"

        # Save suffix to distinguish different map versions
        SAVE_SUFFIX="_map${MAP_NAME}"

        # With augmentation
        train_model "${MODEL_NAME}" "${CONVERT_NAME}" "true" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}" "${SAVE_SUFFIX}"

        # Without augmentation
        train_model "${MODEL_NAME}" "${CONVERT_NAME}" "false" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}" "${SAVE_SUFFIX}"
    done
done

# =============================================================================
# Train Single-frame Models
# =============================================================================

echo ""
echo "=========================================="
echo "🔹 Training Single-frame Models"
echo "=========================================="

for MODEL_INFO in "${SINGLE_FRAME_MODELS[@]}"; do
    MODEL_NAME="${MODEL_INFO%%:*}"
    CONVERT_NAME="${MODEL_INFO##*:}"

    # With augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "true" "" ""

    # Without augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "false" "" ""
done

# =============================================================================
# Train Temporal Models
# =============================================================================

echo ""
echo "=========================================="
echo "🔹 Training Temporal Models"
echo "=========================================="

SEQ_LEN=10
HIDDEN_SIZE=128

for MODEL_INFO in "${TEMPORAL_MODELS[@]}"; do
    MODEL_NAME="${MODEL_INFO%%:*}"
    CONVERT_NAME="${MODEL_INFO##*:}"

    # Hydra overrides for train.py (no dashes, use dots)
    TRAIN_EXTRA="model.seq_len=${SEQ_LEN} model.hidden_size=${HIDDEN_SIZE}"

    # CLI args for convert_weight.py (use dashes)
    CONVERT_EXTRA="--seq-len ${SEQ_LEN} --hidden-size ${HIDDEN_SIZE}"

    # Add --tcn-causal for TCN model (inference uses causal mode)
    if [[ "${MODEL_NAME}" == "TinyLidarNetTCN" ]]; then
        # Training uses tcn_causal=false (from config), but we can pass it explicitly
        # convert uses --tcn-causal only if we want causal inference
        # For now, we keep non-causal to match training
        CONVERT_EXTRA="${CONVERT_EXTRA}"
    fi

    # With augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "true" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}"

    # Without augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "false" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}"
done

# =============================================================================
# Train BEV Models
# =============================================================================

echo ""
echo "=========================================="
echo "🔹 Training BEV Models"
echo "=========================================="

for MODEL_INFO in "${BEV_MODELS[@]}"; do
    MODEL_NAME="${MODEL_INFO%%:*}"
    CONVERT_NAME="${MODEL_INFO##*:}"

    # Hydra overrides for train.py
    TRAIN_EXTRA="model.lane_csv_path='${LANE_CSV_PATH}' model.local_bev_size=${LOCAL_BEV_SIZE} model.local_bev_channels=${LOCAL_BEV_CHANNELS} model.local_resolution=${LOCAL_RESOLUTION} model.global_bev_size=${GLOBAL_BEV_SIZE} model.global_bev_channels=${GLOBAL_BEV_CHANNELS} model.global_resolution=${GLOBAL_RESOLUTION}"

    # CLI args for convert_weight.py
    CONVERT_EXTRA="--local-bev-size ${LOCAL_BEV_SIZE} --local-bev-channels ${LOCAL_BEV_CHANNELS} --global-bev-size ${GLOBAL_BEV_SIZE} --global-bev-channels ${GLOBAL_BEV_CHANNELS}"

    # With augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "true" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}"

    # Without augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "false" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}"
done

# =============================================================================
# Summary
# =============================================================================

echo "" | tee -a "${SUMMARY_FILE}"
echo "=============================================" | tee -a "${SUMMARY_FILE}"
echo "🎉 All Training Complete!" | tee -a "${SUMMARY_FILE}"
echo "Finished: $(date)" | tee -a "${SUMMARY_FILE}"
echo "=============================================" | tee -a "${SUMMARY_FILE}"

echo ""
echo "📂 Checkpoints saved to: ${CHECKPOINT_BASE}/"
ls -la "${CHECKPOINT_BASE}/" 2>/dev/null || echo "   (no checkpoints yet)"

echo ""
echo "📦 Weights saved to: ${WEIGHTS_DIR}/"
ls -la "${WEIGHTS_DIR}/"*.npy 2>/dev/null || echo "   (no weights yet)"

echo ""
echo "📝 Training logs: ${LOG_DIR}/"
echo "📊 Summary: ${SUMMARY_FILE}"
