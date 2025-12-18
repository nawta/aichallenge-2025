#!/bin/bash
# =============================================================================
# TinyLidarNet - Train Map Models Only
# =============================================================================
# For parallel training with BEV models
#
# Usage:
#   ./train_map_models.sh           # Use GPU
#   ./train_map_models.sh --cpu     # Use CPU only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_BASE="${SCRIPT_DIR}/checkpoints"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
LOG_DIR="${SCRIPT_DIR}/training_logs"

INPUT_DIM=1080
EPOCHS=200

USE_CPU=""
if [[ "$1" == "--cpu" ]]; then
    USE_CPU="CUDA_VISIBLE_DEVICES=\"\""
    echo "Running in CPU-only mode"
fi

mkdir -p "${CHECKPOINT_BASE}" "${WEIGHTS_DIR}" "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="${LOG_DIR}/map_training_summary_${TIMESTAMP}.txt"

echo "=============================================" | tee -a "${SUMMARY_FILE}"
echo "TinyLidarNet Map Models Training" | tee -a "${SUMMARY_FILE}"
echo "Started: $(date)" | tee -a "${SUMMARY_FILE}"
echo "=============================================" | tee -a "${SUMMARY_FILE}"

# Map models
MAP_MODELS=("TinyLidarNetMap:tinylidarnet_map")
MAP_IMAGES=("/aichallenge/map_image/1.png" "/aichallenge/map_image/2.png")
MAP_FEATURE_DIM=128

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
    echo "Training: ${FULL_NAME}" | tee -a "${SUMMARY_FILE}"
    echo "   Augmentation: ${AUGMENT}" | tee -a "${SUMMARY_FILE}"
    echo "   Started: $(date)" | tee -a "${SUMMARY_FILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${SUMMARY_FILE}"

    local CMD="${USE_CPU} python3 ${SCRIPT_DIR}/train.py \
        model.name='${MODEL_NAME}' \
        data.augment_mirror=${AUGMENT} \
        train.save_dir='${SAVE_DIR}' \
        train.epochs=${EPOCHS} \
        train.early_stop_patience=null \
        ${TRAIN_EXTRA_ARGS}"

    START_TIME=$(date +%s)

    if eval ${CMD} 2>&1 | tee "${LOG_FILE}"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "Training completed in ${DURATION}s" | tee -a "${SUMMARY_FILE}"

        if [[ -f "${SAVE_DIR}/best_model.pth" ]]; then
            echo "Converting weights..." | tee -a "${SUMMARY_FILE}"
            local CONVERT_CMD="python3 ${SCRIPT_DIR}/convert_weight.py \
                --model ${CONVERT_NAME} \
                --input-dim ${INPUT_DIM} \
                --ckpt '${SAVE_DIR}/best_model.pth' \
                --output '${WEIGHT_FILE}' \
                ${CONVERT_EXTRA_ARGS}"

            if eval ${CONVERT_CMD}; then
                echo "Weights saved: ${WEIGHT_FILE}" | tee -a "${SUMMARY_FILE}"
            else
                echo "Weight conversion failed" | tee -a "${SUMMARY_FILE}"
            fi
        else
            echo "No best_model.pth found" | tee -a "${SUMMARY_FILE}"
        fi
    else
        echo "Training failed! Check log: ${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
    fi
}

echo ""
echo "Training Plan: Map Models"
echo "   Epochs: ${EPOCHS}"
echo "   Models: ${#MAP_MODELS[@]} × ${#MAP_IMAGES[@]} maps × 2 (aug/noaug) = $((${#MAP_MODELS[@]} * ${#MAP_IMAGES[@]} * 2))"
echo ""

for MAP_IMAGE in "${MAP_IMAGES[@]}"; do
    MAP_NAME=$(basename "${MAP_IMAGE}" .png)

    for MODEL_INFO in "${MAP_MODELS[@]}"; do
        MODEL_NAME="${MODEL_INFO%%:*}"
        CONVERT_NAME="${MODEL_INFO##*:}"

        TRAIN_EXTRA="model.map_image_path='${MAP_IMAGE}' model.map_feature_dim=${MAP_FEATURE_DIM}"
        CONVERT_EXTRA="--map-feature-dim ${MAP_FEATURE_DIM}"
        SAVE_SUFFIX="_map${MAP_NAME}"

        train_model "${MODEL_NAME}" "${CONVERT_NAME}" "true" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}" "${SAVE_SUFFIX}"
        train_model "${MODEL_NAME}" "${CONVERT_NAME}" "false" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}" "${SAVE_SUFFIX}"
    done
done

echo "" | tee -a "${SUMMARY_FILE}"
echo "=============================================" | tee -a "${SUMMARY_FILE}"
echo "Map Models Training Complete!" | tee -a "${SUMMARY_FILE}"
echo "Finished: $(date)" | tee -a "${SUMMARY_FILE}"
echo "=============================================" | tee -a "${SUMMARY_FILE}"
