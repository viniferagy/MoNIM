#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/monim/common.sh"
setup_monim_env
prepare_knnlm_layout

BEGIN_DAY="${BEGIN_DAY:-1}"
END_DAY="${END_DAY:-30}"
GPUS="${GPUS:-${REPRO_GPUS:-0,1,2,3}}"
CKPT_NAME="${CKPT_NAME:-gpt2-small}"
DATASET="${DATASET:-daily}"
PT="${PT:-0.0}"
PR="${PR:-0.0}"
FP="${FP:-16}"
USE_COUNT_ACC="${USE_COUNT_ACC:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/monim/final/fullmem_output}"
RESULTS_CSV="${RESULTS_CSV:-${ROOT}/monim/final/fullmem_results_1_30.csv}"
METRIC_GRID_CSV="${METRIC_GRID_CSV:-${ROOT}/monim/final/fullmem_metric_grid_1_30.csv}"
START_STAGGER_SEC="${START_STAGGER_SEC:-2}"

mkdir -p \
  "${ROOT}/monim/final" \
  "${OUTPUT_DIR}/debug/ppl" \
  "${OUTPUT_DIR}/debug/pwd" \
  "${OUTPUT_DIR}/debug/count_acc" \
  "${OUTPUT_DIR}/debug/vars"
ln -sfn "${ROOT}/output/size.json" "${OUTPUT_DIR}/size.json"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "No GPUs configured in GPUS=${GPUS}" >&2
  exit 2
fi

metric_pair_done() {
  local ppl_file="$1"
  local pwd_file="$2"
  local lmbda="$3"
  local temp="$4"
  "${PYTHON_BIN}" "${ROOT}/monim/artifact_status.py" metric-pair-done "${ppl_file}" "${pwd_file}" "${lmbda}" "${temp}"
}

metric_grid_done() {
  local ppl_file="$1"
  local pwd_file="$2"
  "${PYTHON_BIN}" "${ROOT}/monim/artifact_status.py" metric-grid-done "${ppl_file}" "${pwd_file}"
}

preflight() {
  "${PYTHON_BIN}" - <<'PY'
import faiss
import torch

if not torch.cuda.is_available():
    raise SystemExit("Torch CUDA is not available")
if torch.cuda.device_count() < 1:
    raise SystemExit("No CUDA devices visible")
res = faiss.StandardGpuResources()
print(f"preflight ok: torch_cuda_devices={torch.cuda.device_count()} faiss_gpu_resources={type(res).__name__}")
PY
}

eval_pair() {
  local day="$1"
  local lmbda="$2"
  local temp="$3"
  local gpu="$4"
  local param="${PT},${PR}"
  local total_prefix="${CKPT_NAME}_${day}__fp${FP}_p_${param}"
  local text_bin="${ROOT}/data/knnlm/datasets/${DATASET}/${day}/bin"
  local ckpt="${ROOT}/data/knnlm/checkpoints/${CKPT_NAME}/checkpoint_best.pt"
  local dstore="${ROOT}/data/knnlm/storage/${CKPT_NAME}/${DATASET}/dstore/${total_prefix}"
  local index="${ROOT}/data/knnlm/storage/${CKPT_NAME}/${DATASET}/knn/${total_prefix}.index"
  local count_args=()

  if [ "${USE_COUNT_ACC}" = "1" ]; then
    count_args=(--count-acc --deviation)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT}/knnlm/eval_lm.py" "${text_bin}" \
    --path "${ckpt}" \
    --sample-break-mode none --max-tokens 1024 \
    --softmax-batch 128 --gen-subset test \
    --context-window 512 \
    --infer-dstore-path "${dstore}" --index-file "${index}" --knn-keytype last_ffn_input \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --knnlm --fp16 --no-load-keys --knn-sim-func do_not_recomp_l2 --k 1024 --probe 32 \
    --gpu-index True \
    --prune-threshold "${PT}" --prune-rate "${PR}" \
    --stop-size "" \
    --lmbda "${lmbda}" --knn-temp "${temp}" \
    --output-dir "${OUTPUT_DIR}" \
    --dstore-fp16 "${count_args[@]}" \
    --ckpt-name "${CKPT_NAME}" --date "${day}"
}

check_day_artifacts() {
  local day="$1"
  local param="${PT},${PR}"
  local total_prefix="${CKPT_NAME}_${day}__fp${FP}_p_${param}"
  local text_bin="${ROOT}/data/knnlm/datasets/${DATASET}/${day}/bin"
  local dstore="${ROOT}/storage/${CKPT_NAME}/${DATASET}/dstore/${total_prefix}"
  local index="${ROOT}/storage/${CKPT_NAME}/${DATASET}/knn/${total_prefix}.index"

  [ -d "${text_bin}" ] || { echo "Missing text bin: ${text_bin}" >&2; exit 1; }
  [ -e "${dstore}_vals.npy" ] || { echo "Missing FullMem datastore vals: ${dstore}_vals.npy" >&2; exit 1; }
  [ -e "${index}" ] || { echo "Missing FullMem FAISS index: ${index}" >&2; exit 1; }
}

run_day() {
  local day="$1"
  local param="${PT},${PR}"
  local prefix="${CKPT_NAME}_${day}__fp${FP}_p_${param}.txt"
  local ppl_file="${OUTPUT_DIR}/debug/ppl/${prefix}"
  local pwd_file="${OUTPUT_DIR}/debug/pwd/${prefix}"
  local jobs=()
  local gpu_idx=0
  local status=0

  check_day_artifacts "${day}"
  if metric_grid_done "${ppl_file}" "${pwd_file}"; then
    echo "== FullMem day ${day}: complete metric grid; skip"
    return 0
  fi

  echo "== FullMem day ${day}: infer"
  for lmbda in $(seq 0.4 0.05 0.6); do
    for temp in $(seq 8.5 0.5 15.5); do
      if metric_pair_done "${ppl_file}" "${pwd_file}" "${lmbda}" "${temp}"; then
        echo "== FullMem day ${day}: skip lmbda=${lmbda} temp=${temp}"
        continue
      fi
      gpu="${GPU_LIST[$gpu_idx]}"
      eval_pair "${day}" "${lmbda}" "${temp}" "${gpu}" &
      jobs+=("$!")
      if [ "${START_STAGGER_SEC}" != "0" ]; then
        sleep "${START_STAGGER_SEC}"
      fi
      gpu_idx=$(( (gpu_idx + 1) % ${#GPU_LIST[@]} ))
      if [ "${#jobs[@]}" -ge "${#GPU_LIST[@]}" ]; then
        if ! wait "${jobs[0]}"; then
          status=1
        fi
        jobs=("${jobs[@]:1}")
      fi
    done
  done

  for job in "${jobs[@]}"; do
    if ! wait "${job}"; then
      status=1
    fi
  done

  if [ "${status}" -ne 0 ]; then
    echo "One or more infer workers failed for FullMem day ${day}; retrying missing metric pairs serially" >&2
    status=0
    for lmbda in $(seq 0.4 0.05 0.6); do
      for temp in $(seq 8.5 0.5 15.5); do
        if metric_pair_done "${ppl_file}" "${pwd_file}" "${lmbda}" "${temp}"; then
          continue
        fi
        if ! eval_pair "${day}" "${lmbda}" "${temp}" "${GPU_LIST[0]}"; then
          status=1
        fi
      done
    done
    if [ "${status}" -ne 0 ]; then
      echo "Serial infer retries failed for FullMem day ${day}" >&2
      exit "${status}"
    fi
  fi

  metric_grid_done "${ppl_file}" "${pwd_file}"
}

preflight

for day in $(seq "${BEGIN_DAY}" 1 "${END_DAY}"); do
  run_day "${day}"
done

"${PYTHON_BIN}" "${ROOT}/monim/check_metric_grid.py" \
  --begin "${BEGIN_DAY}" --end "${END_DAY}" \
  --output-dir "${OUTPUT_DIR}" --methods fullmem --strict \
  > "${METRIC_GRID_CSV}"

"${PYTHON_BIN}" "${ROOT}/monim/summarize_results.py" \
  --begin "${BEGIN_DAY}" --end "${END_DAY}" \
  --output-dir "${OUTPUT_DIR}" --methods fullmem --strict \
  > "${RESULTS_CSV}"

echo "FullMem results: ${RESULTS_CSV}"
echo "FullMem metric grid: ${METRIC_GRID_CSV}"
