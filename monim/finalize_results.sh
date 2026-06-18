#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/monim/common.sh"
setup_monim_env

BEGIN_DAY="${BEGIN_DAY:-1}"
END_DAY="${END_DAY:-30}"
OUT_DIR="${OUT_DIR:-${ROOT}/monim/final}"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" "${ROOT}/monim/audit_artifacts.py" \
  --begin "${BEGIN_DAY}" --end "${END_DAY}" \
  > "${OUT_DIR}/artifact_audit_${BEGIN_DAY}_${END_DAY}.csv"

"${PYTHON_BIN}" "${ROOT}/monim/default_results.py" \
  --begin "${BEGIN_DAY}" --end "${END_DAY}" --out-dir "${OUT_DIR}" --strict

cat <<EOF
Final validation passed.
Artifacts audit: ${OUT_DIR}/artifact_audit_${BEGIN_DAY}_${END_DAY}.csv
Metric grid:     ${OUT_DIR}/metric_grid_${BEGIN_DAY}_${END_DAY}.csv
Results:         ${OUT_DIR}/results_${BEGIN_DAY}_${END_DAY}.csv
EOF
