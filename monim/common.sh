#!/usr/bin/env bash

monim_root() {
  cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd
}

setup_monim_env() {
  ROOT="${ROOT:-$(monim_root)}"
  KNNLM_VENV="${KNNLM_VENV:-${ROOT}/.venv-uv}"
  PYTHON_BIN="${PYTHON_BIN:-${KNNLM_VENV}/bin/python}"

  export ROOT
  export KNNLM_VENV
  export PYTHON_BIN
  export PATH="${KNNLM_VENV}/bin:${PATH}"
  export PYTHONPATH="${ROOT}/knnlm:${ROOT}/monim:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}

prepare_knnlm_layout() {
  mkdir -p "${ROOT}/data/knnlm" "${ROOT}/storage" "${ROOT}/output/debug"
  ln -sfn ../../datasets "${ROOT}/data/knnlm/datasets"
  ln -sfn ../../checkpoints "${ROOT}/data/knnlm/checkpoints"
  ln -sfn ../../output "${ROOT}/data/knnlm/output"
  ln -sfn ../../storage "${ROOT}/data/knnlm/storage"
}

prepare_log_dir() {
  LOG_DIR="${LOG_DIR:-${ROOT}/monim/logs}"
  export LOG_DIR
  mkdir -p "${LOG_DIR}"
}
