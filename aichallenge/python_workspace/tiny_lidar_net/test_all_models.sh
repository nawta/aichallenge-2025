#!/bin/bash
# =============================================================================
# TinyLidarNet - Quick Test Script for All Models
# =============================================================================
# Tests all model architectures with 1 epoch to verify they work correctly.
#
# Usage (inside Docker container):
#   cd /aichallenge/python_workspace/tiny_lidar_net
#   ./test_all_models.sh
# =============================================================================

set -e  # Exit on error

# Navigate to script directory (tiny_lidar_net)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Test parameters
TEST_EPOCHS=1
TEST_BATCH_SIZE=8

echo "============================================="
echo "TinyLidarNet - Quick Model Test"
echo "Epochs: ${TEST_EPOCHS}, Batch Size: ${TEST_BATCH_SIZE}"
echo "============================================="

# Function to test a model
test_model() {
    local MODEL_NAME=$1
    local EXTRA_ARGS=$2

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: ${MODEL_NAME}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if python3 train.py \
        model.name="${MODEL_NAME}" \
        train.epochs=${TEST_EPOCHS} \
        train.batch_size=${TEST_BATCH_SIZE} \
        train.save_dir="./tmp/test_${MODEL_NAME}" \
        train.log_dir="./tmp/logs_${MODEL_NAME}" \
        data.augment_mirror=false \
        ${EXTRA_ARGS} 2>&1 | head -100; then
        echo "✅ ${MODEL_NAME}: PASSED"
        return 0
    else
        echo "❌ ${MODEL_NAME}: FAILED"
        return 1
    fi
}

# Create tmp directory
mkdir -p ./tmp

PASSED=0
FAILED=0

# =============================================================================
# Test Single-frame Models
# =============================================================================
echo ""
echo "=========================================="
echo "Testing Single-frame Models"
echo "=========================================="

for MODEL in "TinyLidarNet" "TinyLidarNetSmall" "TinyLidarNetDeep" "TinyLidarNetFusion"; do
    if test_model "${MODEL}"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

# =============================================================================
# Test Temporal Models
# =============================================================================
echo ""
echo "=========================================="
echo "Testing Temporal Models"
echo "=========================================="

for MODEL in "TinyLidarNetStacked" "TinyLidarNetBiLSTM" "TinyLidarNetTCN"; do
    EXTRA="model.seq_len=5 model.hidden_size=64"
    if test_model "${MODEL}" "${EXTRA}"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

# =============================================================================
# Test Map Model
# =============================================================================
echo ""
echo "=========================================="
echo "Testing Map Model"
echo "=========================================="

MAP_PATH="/aichallenge/map_image/1.png"
if [ -f "${MAP_PATH}" ]; then
    EXTRA="model.map_image_path='${MAP_PATH}' model.map_feature_dim=64"
    if test_model "TinyLidarNetMap" "${EXTRA}"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
else
    echo "⚠️  Map image not found: ${MAP_PATH}, skipping TinyLidarNetMap"
fi

# =============================================================================
# Test BEV Models
# =============================================================================
echo ""
echo "=========================================="
echo "Testing BEV Models"
echo "=========================================="

LANE_CSV="/aichallenge/workspace/src/aichallenge_submit/laserscan_generator/map/lane.csv"
if [ -f "${LANE_CSV}" ]; then
    BEV_EXTRA="model.lane_csv_path='${LANE_CSV}' model.local_bev_size=32 model.global_bev_size=64"

    for MODEL in "TinyLidarNetLocalBEV" "TinyLidarNetGlobalBEV" "TinyLidarNetDualBEV"; do
        if test_model "${MODEL}" "${BEV_EXTRA}"; then
            ((PASSED++))
        else
            ((FAILED++))
        fi
    done
else
    echo "⚠️  Lane CSV not found: ${LANE_CSV}, skipping BEV models"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================="
echo "Test Summary"
echo "============================================="
echo "✅ Passed: ${PASSED}"
echo "❌ Failed: ${FAILED}"
echo ""

# Cleanup
echo "Cleaning up test files..."
rm -rf ./tmp

if [ ${FAILED} -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    echo "⚠️  Some tests failed. Check output above."
    exit 1
fi
