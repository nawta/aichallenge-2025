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

# =============================================================================
# Training Function
# =============================================================================

train_model() {
    local MODEL_NAME=$1
    local CONVERT_NAME=$2
    local AUGMENT=$3  # "true" or "false"
    local EXTRA_ARGS=$4
    
    local AUG_SUFFIX=""
    if [[ "$AUGMENT" == "true" ]]; then
        AUG_SUFFIX="_aug"
    else
        AUG_SUFFIX="_noaug"
    fi
    
    local SAVE_DIR="${CHECKPOINT_BASE}/${MODEL_NAME}${AUG_SUFFIX}"
    local LOG_FILE="${LOG_DIR}/${MODEL_NAME}${AUG_SUFFIX}_${TIMESTAMP}.log"
    local WEIGHT_FILE="${WEIGHTS_DIR}/${MODEL_NAME}${AUG_SUFFIX}.npy"
    
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${SUMMARY_FILE}"
    echo "🚀 Training: ${MODEL_NAME}${AUG_SUFFIX}" | tee -a "${SUMMARY_FILE}"
    echo "   Augmentation: ${AUGMENT}" | tee -a "${SUMMARY_FILE}"
    echo "   Save Dir: ${SAVE_DIR}" | tee -a "${SUMMARY_FILE}"
    echo "   Started: $(date)" | tee -a "${SUMMARY_FILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${SUMMARY_FILE}"
    
    # Build training command
    local CMD="${USE_CPU} python3 ${SCRIPT_DIR}/train.py \
        model.name='${MODEL_NAME}' \
        data.augment_mirror=${AUGMENT} \
        train.save_dir='${SAVE_DIR}' \
        ${EXTRA_ARGS}"
    
    # Run training
    START_TIME=$(date +%s)
    
    if eval ${CMD} 2>&1 | tee "${LOG_FILE}"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "✅ Training completed in ${DURATION}s" | tee -a "${SUMMARY_FILE}"
        
        # Convert weights
        if [[ -f "${SAVE_DIR}/best_model.pth" ]]; then
            echo "📦 Converting weights..." | tee -a "${SUMMARY_FILE}"
            
            local CONVERT_CMD="python3 ${SCRIPT_DIR}/convert_weight.py \
                --model ${CONVERT_NAME} \
                --ckpt '${SAVE_DIR}/best_model.pth' \
                --output '${WEIGHT_FILE}' \
                ${EXTRA_ARGS}"
            
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

echo ""
echo "📋 Training Plan:"
echo "   Single-frame models: ${#SINGLE_FRAME_MODELS[@]} × 2 (aug/noaug)"
echo "   Temporal models: ${#TEMPORAL_MODELS[@]} × 2 (aug/noaug)"
echo "   Total: $((( ${#SINGLE_FRAME_MODELS[@]} + ${#TEMPORAL_MODELS[@]} ) * 2)) training runs"
echo ""

# Train single-frame models
for MODEL_INFO in "${SINGLE_FRAME_MODELS[@]}"; do
    MODEL_NAME="${MODEL_INFO%%:*}"
    CONVERT_NAME="${MODEL_INFO##*:}"
    
    # With augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "true" ""
    
    # Without augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "false" ""
done

# Train temporal models
SEQ_LEN=10
HIDDEN_SIZE=128

for MODEL_INFO in "${TEMPORAL_MODELS[@]}"; do
    MODEL_NAME="${MODEL_INFO%%:*}"
    CONVERT_NAME="${MODEL_INFO##*:}"
    
    EXTRA_ARGS="model.seq_len=${SEQ_LEN} model.hidden_size=${HIDDEN_SIZE} --seq-len ${SEQ_LEN} --hidden-size ${HIDDEN_SIZE}"
    
    # With augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "true" "${EXTRA_ARGS}"
    
    # Without augmentation
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "false" "${EXTRA_ARGS}"
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
