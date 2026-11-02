#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/spo_ddi_5fold}"
MOLFORMER_FEATURE_PATH="${MOLFORMER_FEATURE_PATH:-trimnet/data/ddinter/molformer_ibm_molformer_xl_both_10pct.npz}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results"
cd "${ROOT}"

for fold in 0 1 2 3 4; do
  log_file="${RUN_ROOT}/logs/fold_${fold}.log"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u main_molformer_pairgate_ordinal.py \
    --dataset ddinter \
    --zhongzi "${fold}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH_SIZE}" \
    --molformer_feature_path "${MOLFORMER_FEATURE_PATH}" \
    --checkpoint_root "${RUN_ROOT}/checkpoints" \
    --result_root "${RUN_ROOT}/results" \
    --out_file "${RUN_ROOT}/spo_ddi_result.txt" \
    --auto_resume \
    2>&1 | tee "${log_file}"
done
