#!/bin/bash
# =============================================================================
# TinyLidarNet - Train BEV Models Only
# =============================================================================
# For parallel training with Map models
#
# Usage:
#   ./train_bev_models.sh           # Use GPU
#   ./train_bev_models.sh --cpu     # Use CPU only
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
SUMMARY_FILE="${LOG_DIR}/bev_training_summary_${TIMESTAMP}.txt"

echo "=============================================" | tee -a "${SUMMARY_FILE}"
echo "TinyLidarNet BEV Models Training" | tee -a "${SUMMARY_FILE}"
echo "Started: $(date)" | tee -a "${SUMMARY_FILE}"
echo "=============================================" | tee -a "${SUMMARY_FILE}"

# BEV models
BEV_MODELS=(
    "TinyLidarNetLocalBEV:tinylidarnet_local_bev"
    "TinyLidarNetGlobalBEV:tinylidarnet_global_bev"
    "TinyLidarNetDualBEV:tinylidarnet_dual_bev"
)

LANE_CSV_PATH="/aichallenge/workspace/src/aichallenge_submit/laserscan_generator/map/lane.csv"
LOCAL_BEV_SIZE=64
LOCAL_BEV_CHANNELS=2
LOCAL_RESOLUTION=1.0
GLOBAL_BEV_SIZE=128
GLOBAL_BEV_CHANNELS=3
GLOBAL_RESOLUTION=1.5

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
echo "Training Plan: BEV Models"
echo "   Epochs: ${EPOCHS}"
echo "   Models: ${#BEV_MODELS[@]} × 2 (aug/noaug) = $((${#BEV_MODELS[@]} * 2))"
echo ""

for MODEL_INFO in "${BEV_MODELS[@]}"; do
    MODEL_NAME="${MODEL_INFO%%:*}"
    CONVERT_NAME="${MODEL_INFO##*:}"

    TRAIN_EXTRA="model.lane_csv_path='${LANE_CSV_PATH}' model.local_bev_size=${LOCAL_BEV_SIZE} model.local_bev_channels=${LOCAL_BEV_CHANNELS} model.local_resolution=${LOCAL_RESOLUTION} model.global_bev_size=${GLOBAL_BEV_SIZE} model.global_bev_channels=${GLOBAL_BEV_CHANNELS} model.global_resolution=${GLOBAL_RESOLUTION}"
    CONVERT_EXTRA="--local-bev-size ${LOCAL_BEV_SIZE} --local-bev-channels ${LOCAL_BEV_CHANNELS} --global-bev-size ${GLOBAL_BEV_SIZE} --global-bev-channels ${GLOBAL_BEV_CHANNELS}"

    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "true" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}"
    train_model "${MODEL_NAME}" "${CONVERT_NAME}" "false" "${TRAIN_EXTRA}" "${CONVERT_EXTRA}"
done

echo "" | tee -a "${SUMMARY_FILE}"
echo "=============================================" | tee -a "${SUMMARY_FILE}"
echo "BEV Models Training Complete!" | tee -a "${SUMMARY_FILE}"
echo "Finished: $(date)" | tee -a "${SUMMARY_FILE}"
echo "=============================================" | tee -a "${SUMMARY_FILE}"
