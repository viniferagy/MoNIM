#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/monim/common.sh"
setup_monim_env
prepare_log_dir

bash "${ROOT}/monim/eval_fullmem_1_30.sh" \
  2>&1 | tee "${LOG_DIR}/fullmem_1_30.$(date +%Y%m%d_%H%M%S).log"
