#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/monim/common.sh"
setup_monim_env
prepare_log_dir

bash "${ROOT}/monim/run_cl_1_30.sh" \
  --methods fullmem \
  --begin 1 \
  --end 30 \
  --gpus "${REPRO_GPUS:-0,1,2,3}" \
  --no-eval \
  2>&1 | tee "${LOG_DIR}/cl_fullmem_artifacts_1_30.$(date +%Y%m%d_%H%M%S).log"
bash "${ROOT}/monim/eval_fullmem_1_30.sh" \
  2>&1 | tee "${LOG_DIR}/fullmem_1_30.$(date +%Y%m%d_%H%M%S).log"
bash "${ROOT}/monim/run_cl_1_30.sh" \
  --methods monim \
  --begin 1 \
  --end 30 \
  --gpus "${REPRO_GPUS:-0,1,2,3}" \
  --eval-each-day \
  2>&1 | tee "${LOG_DIR}/cl_1_30.$(date +%Y%m%d_%H%M%S).log"
bash "${ROOT}/monim/finalize_results.sh"
