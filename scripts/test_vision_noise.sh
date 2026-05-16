#!/bin/bash
# Test RL policy robustness to hole position sensing error.
# Runs evaluation at multiple vision noise levels and reports success rates.
#
# Usage:
#   bash scripts/test_vision_noise.sh <assembly_id> <checkpoint_path> [num_envs] [num_eval_trials]
#
# Example:
#   bash scripts/test_vision_noise.sh 01036 checkpoints/00783.pth 128 256

set -euo pipefail

ASSEMBLY_ID="${1:?Usage: $0 <assembly_id> <checkpoint_path> [num_envs]}"
CHECKPOINT="${2:?Usage: $0 <assembly_id> <checkpoint_path> [num_envs]}"
NUM_ENVS="${3:-128}"
NUM_EVAL_TRIALS="${4:-256}"

# Noise levels to test (in meters): 0, 5mm, 10mm, 15mm, 20mm
NOISE_LEVELS=("0.0" "0.005" "0.010" "0.015" "0.020")
JITTER_RATIO="${VISION_JITTER_RATIO:-0.2}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRSA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/../source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py"

cd "${SRSA_ROOT}"

echo "============================================"
echo "Vision Noise Robustness Test"
echo "Assembly ID: ${ASSEMBLY_ID}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Num Envs:    ${NUM_ENVS}"
echo "Eval Trials: ${NUM_EVAL_TRIALS}"
echo "Jitter Ratio:${JITTER_RATIO}"
echo "============================================"
echo ""

for NOISE in "${NOISE_LEVELS[@]}"; do
    NOISE_MM=$(echo "${NOISE} * 1000" | bc)
    JITTER=$(echo "${NOISE} * ${JITTER_RATIO}" | bc -l)
    EVAL_FILE="evaluation_${ASSEMBLY_ID}_noise_${NOISE_MM}mm.h5"

    echo "--- Testing vision noise: ${NOISE}m (${NOISE_MM}mm), jitter ${JITTER}m ---"

    python "${RUN_SCRIPT}" \
        --assembly_id "${ASSEMBLY_ID}" \
        --checkpoint "${CHECKPOINT}" \
        --vision_noise "${NOISE}" \
        --vision_jitter "${JITTER}" \
        --sparse --no_sbc \
        --log_eval \
        --headless \
        --num_envs "${NUM_ENVS}" \
        --num_eval_trials "${NUM_EVAL_TRIALS}"

    # The eval writes to evaluation_<assembly_id>.h5, rename it
    DEFAULT_EVAL="evaluation_${ASSEMBLY_ID}.h5"
    if [ -f "${DEFAULT_EVAL}" ]; then
        mv "${DEFAULT_EVAL}" "${EVAL_FILE}"
    fi

    # Parse success rate from HDF5
    python -c "
import h5py, numpy as np, sys
try:
    with h5py.File('${EVAL_FILE}', 'r') as f:
        success = f['success'][:]
        rate = np.mean(success) * 100
        count = int(np.sum(success))
        total = len(success)
        print(f'  Success rate: {rate:.1f}% ({count}/{total})')
except Exception as e:
    print(f'  Error reading results: {e}')
"

    echo ""
done

echo "============================================"
echo "Summary:"
echo "============================================"
printf "%-15s %s\n" "Noise (mm)" "Success Rate"
printf "%-15s %s\n" "----------" "------------"
for NOISE in "${NOISE_LEVELS[@]}"; do
    NOISE_MM=$(echo "${NOISE} * 1000" | bc)
    EVAL_FILE="evaluation_${ASSEMBLY_ID}_noise_${NOISE_MM}mm.h5"
    if [ -f "${EVAL_FILE}" ]; then
        RESULT=$(python -c "
import h5py, numpy as np
with h5py.File('${EVAL_FILE}', 'r') as f:
    success = f['success'][:]
    rate = np.mean(success) * 100
    count = int(np.sum(success))
    total = len(success)
    print(f'{rate:.1f}% ({count}/{total})')
")
        printf "%-15s %s\n" "${NOISE_MM}" "${RESULT}"
    else
        printf "%-15s %s\n" "${NOISE_MM}" "N/A"
    fi
done
