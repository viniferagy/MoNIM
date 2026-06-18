#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/monim/common.sh"
setup_monim_env
prepare_knnlm_layout

SCRIPT_DIR="${ROOT}/knnlm/script"

METHODS="monim"
BEGIN_DAY=1
END_DAY=30
GPUS="0,1,2,3"
CKPT_NAME="gpt2-small"
DATASET="daily"
DATA_TYPE=""
USE_K=10
EVAL=1
EVAL_EACH_DAY=0
EVAL_ONLY=0
REUSE_NET=0
FAISS_CPU_TRAIN="${KNNLM_FAISS_TRAIN_ON_CPU:-0}"

usage() {
  cat <<'USAGE'
Usage: bash monim/run_cl_1_30.sh [options]

Options:
  --methods monim           Methods to run with the adapter pipeline.
  --begin N                 First day to run. Default: 1.
  --end N                   Last day to run. Default: 30.
  --gpus LIST               CUDA_VISIBLE_DEVICES-style list. Default: 0,1,2,3.
  --ckpt-name NAME          Checkpoint name. Default: gpt2-small.
  --eval-each-day           Run inference after every day, not only after --end.
  --eval-only               Skip datastore/index/net building and run inference only.
  --reuse-net               Reuse existing MoNIM adapter checkpoints; skip feature export and net training.
  --cpu-train-index         Train FAISS IVF-PQ centroids on CPU instead of GPU.
  --no-eval                 Skip final inference.
  -h, --help                Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods) METHODS="$2"; shift 2 ;;
    --begin) BEGIN_DAY="$2"; shift 2 ;;
    --end) END_DAY="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --ckpt-name) CKPT_NAME="$2"; shift 2 ;;
    --eval-each-day) EVAL_EACH_DAY=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --reuse-net) REUSE_NET=1; shift ;;
    --cpu-train-index) FAISS_CPU_TRAIN=1; shift ;;
    --no-eval) EVAL=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

gpu_count() {
  local csv="$1"
  local gpu count=0
  IFS=',' read -r -a gpus <<< "${csv}"
  for gpu in "${gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -n "${gpu}" ]] && count=$((count + 1))
  done
  echo "${count}"
}

run_pipe() {
  (cd "${SCRIPT_DIR}" && bash ./pipe.sh "$@")
}

run_multi() {
  local action="$1"
  local command="$2"
  local gpus="$3"
  (cd "${SCRIPT_DIR}" && "${PYTHON_BIN}" multi.py --action "${action}" --command "${command}" --gpu "${gpus}")
}

gpu_array() {
  local csv="$1"
  local gpu
  IFS=',' read -r -a gpus <<< "${csv}"
  for gpu in "${gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -n "${gpu}" ]] && echo "${gpu}"
  done
}

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

json_size_entry() {
  local day="$1"
  local param="$2"
  "${PYTHON_BIN}" "${ROOT}/monim/artifact_status.py" size-entry "${ROOT}/output/size.json" "${DATASET}" "${CKPT_NAME}" "${day}" "${param}"
}

new_dstore_size() {
  local day="$1"
  local param="$2"
  json_size_entry "${day}" "${param}-now"
}

merged_dstore_size() {
  local day="$1"
  local param="$2"
  json_size_entry "${day}" "${param}"
}

artifact_prefix() {
  local day="$1"
  local param="$2"
  echo "${CKPT_NAME}_${DATA_TYPE}${day}__fp16_p_${param}"
}

dstore_path() {
  local day="$1"
  local param="$2"
  echo "${ROOT}/storage/${CKPT_NAME}/${DATASET}/dstore/$(artifact_prefix "${day}" "${param}")"
}

index_path() {
  local day="$1"
  local param="$2"
  echo "${ROOT}/storage/${CKPT_NAME}/${DATASET}/knn/$(artifact_prefix "${day}" "${param}").index"
}

net_feature_path() {
  local day="$1"
  local pt="$2"
  local pr="$3"
  echo "${ROOT}/datasets/${DATASET}/${DATA_TYPE}${day}/net/total/${CKPT_NAME}/${pt}_${pr}"
}

dstore_merged_done() {
  local day="$1"
  local param="$2"
  local dstore size
  dstore="$(dstore_path "${day}" "${param}")"
  size="$(merged_dstore_size "${day}" "${param}" 2>/dev/null || true)"
  [[ -n "${size}" && "${size}" -gt 0 && -s "${dstore}_keys.npy" && -s "${dstore}_vals.npy" ]]
}

today_shards_done() {
  local day="$1"
  local param="$2"
  local shards="$3"
  local dstore expected total i keys vals count
  dstore="$(dstore_path "${day}" "${param}")"
  expected="$(new_dstore_size "${day}" "${param}" 2>/dev/null || true)"
  [[ -n "${expected}" && "${expected}" -gt 0 ]] || return 1

  total=0
  for i in $(seq 0 $((shards - 1))); do
    keys="${dstore}_keys.today.${i}.npy"
    vals="${dstore}_vals.today.${i}.npy"
    [[ -s "${keys}" && -s "${vals}" ]] || return 1
    count="$("${PYTHON_BIN}" "${ROOT}/monim/artifact_status.py" npy-count "${keys}" "${vals}")" || return 1
    total=$((total + count))
  done
  [[ "${total}" -eq "${expected}" ]]
}

index_trained_done() {
  local day="$1"
  local param="$2"
  local index
  index="$(index_path "${day}" "${param}")"
  [[ -s "${index}.trained" ]]
}

index_shards_done() {
  local day="$1"
  local param="$2"
  local shards="$3"
  local index i
  index="$(index_path "${day}" "${param}")"
  for i in $(seq 0 $((shards - 1))); do
    [[ -s "${index}.${i}" ]] || return 1
  done
}

index_done() {
  local day="$1"
  local param="$2"
  local index
  index="$(index_path "${day}" "${param}")"
  [[ -s "${index}" ]]
}

features_done() {
  local day="$1"
  local pt="$2"
  local pr="$3"
  local shards="$4"
  local feature_dir i
  feature_dir="$(net_feature_path "${day}" "${pt}" "${pr}")"
  for i in $(seq 0 $((shards - 1))); do
    [[ -e "${feature_dir}/train_ctxt.${i}.jsonl" && -e "${feature_dir}/train_others.${i}.jsonl" ]] || return 1
    [[ -e "${feature_dir}/test_ctxt.${i}.jsonl" && -e "${feature_dir}/test_others.${i}.jsonl" ]] || return 1
  done
}

checkpoint_newer_than_inputs() {
  local ckpt="$1"
  shift
  [[ -s "${ckpt}" ]] || return 1
  "${PYTHON_BIN}" "${ROOT}/monim/artifact_status.py" checkpoint-newer "${ckpt}" "$@"
}

method_params() {
  "${PYTHON_BIN}" "${ROOT}/monim/artifact_status.py" method-params "$1"
}

method_uses_adapter() {
  "${PYTHON_BIN}" "${ROOT}/monim/artifact_status.py" method-uses-adapter "$1"
}

net_checkpoint() {
  local day="$1"
  local pt="$2"
  local pr="$3"
  local net_param="${pt}_${pr}"
  echo "${ROOT}/datasets/${DATASET}/${DATA_TYPE}${day}/net/total/${CKPT_NAME}/${net_param}/checkpoint/metak.l1.0.05.ngram0.hid128.nl4.bs64.drop0.2.ftall.seed927.use_k${USE_K}.metadim1/checkpoint_best.pt"
}

run_day() {
  local method="$1"
  local day="$2"
  local pt="$3"
  local pr="$4"
  local shards
  shards="$(gpu_count "${GPUS}")"

  local command_common="--ckpt-name ${CKPT_NAME} --dataset ${DATASET} --data-type ${DATA_TYPE:-None} --date ${day} --pt ${pt} --pr ${pr} --dstore-fp16 --use-k ${USE_K} --num-shards ${shards}"
  local command_day="${command_common}"
  local command_save="${command_common}"
  local param="${pt},${pr}"
  local dstore index feature_dir
  dstore="$(dstore_path "${day}" "${param}")"
  index="$(index_path "${day}" "${param}")"
  feature_dir="$(net_feature_path "${day}" "${pt}" "${pr}")"
  if [[ "${day}" -eq 1 ]]; then
    command_day="${command_day} --continual False"
    command_save="${command_day}"
  elif method_uses_adapter "${method}"; then
    command_save="${command_save} --net"
  fi

  if dstore_merged_done "${day}" "${param}"; then
    echo "== ${method} day ${day}: dstore complete; skip save/merge (${dstore})"
  else
    if today_shards_done "${day}" "${param}" "${shards}"; then
      echo "== ${method} day ${day}: save shards complete; skip save"
    else
      echo "== ${method} day ${day}: save"
      run_multi save "${command_save}" "${GPUS}"
    fi

    echo "== ${method} day ${day}: merge"
    run_pipe --action merge ${command_day} --gpu "${GPUS}" --shard-id 0
  fi

  if index_trained_done "${day}" "${param}"; then
    echo "== ${method} day ${day}: trained index complete; skip build_index"
  else
    echo "== ${method} day ${day}: build_index"
    KNNLM_FAISS_TRAIN_ON_CPU="${FAISS_CPU_TRAIN}" run_pipe --action build_index ${command_day} --gpu "${GPUS}" --shard-id 0
  fi

  if index_done "${day}" "${param}"; then
    echo "== ${method} day ${day}: index complete; skip add_index/merge_index (${index})"
  else
    if index_shards_done "${day}" "${param}" "${shards}"; then
      echo "== ${method} day ${day}: index shards complete; skip add_index"
    else
      echo "== ${method} day ${day}: add_index"
      run_multi add_index "${command_day}" "${GPUS}"
    fi

    echo "== ${method} day ${day}: merge_index"
    run_pipe --action merge_index ${command_day} --gpu "${GPUS}" --shard-id 0
  fi

  if ! method_uses_adapter "${method}"; then
    echo "== ${method} day ${day}: no adapter stages"
    return
  fi

  local net_ckpt
  net_ckpt="$(net_checkpoint "${day}" "${pt}" "${pr}")"
  if [[ "${REUSE_NET}" -eq 1 && -s "${net_ckpt}" ]]; then
    echo "== ${method} day ${day}: reuse_net (${net_ckpt})"
    return
  fi
  if checkpoint_newer_than_inputs "${net_ckpt}" "${dstore}_vals.npy" "${index}"; then
    echo "== ${method} day ${day}: train_net complete; skip (${net_ckpt})"
    return
  fi
  if [[ "${REUSE_NET}" -eq 1 ]]; then
    echo "Missing reusable net checkpoint: ${net_ckpt}" >&2
    exit 1
  fi

  if features_done "${day}" "${pt}" "${pr}" "${shards}"; then
    echo "== ${method} day ${day}: features complete; skip build_features/save_features (${feature_dir})"
  else
    echo "== ${method} day ${day}: build_features"
    run_pipe --action build_features ${command_day} --gpu "${GPUS}" --shard-id 0
    echo "== ${method} day ${day}: save_features"
    run_multi save_features "${command_day}" "${GPUS}"
  fi

  echo "== ${method} day ${day}: train_net"
  run_pipe --action train_net ${command_day} --gpu "${GPUS}" --shard-id 0
}

run_eval() {
  local method="$1"
  local day="$2"
  local pt="$3"
  local pr="$4"
  local param="${pt},${pr}"
  local prefix="${CKPT_NAME}_${day}__fp16_p_${param}.txt"
  local command_common="--ckpt-name ${CKPT_NAME} --dataset ${DATASET} --data-type ${DATA_TYPE:-None} --date ${day} --pt ${pt} --pr ${pr} --dstore-fp16 --count-acc --deviation --use-k ${USE_K}"
  local ppl_file="${ROOT}/output/debug/ppl/${prefix}"
  local pwd_file="${ROOT}/output/debug/pwd/${prefix}"

  if method_uses_adapter "${method}"; then
    command_common="${command_common} --net"
  fi

  echo "== ${method} day ${day}: infer"

  mapfile -t gpus < <(gpu_array "${GPUS}")

  local jobs=()
  local gpu_idx=0
  local status=0
  for lmbda in $(seq 0.4 0.05 0.6); do
    for temp in $(seq 8.5 0.5 15.5); do
      if metric_pair_done "${ppl_file}" "${pwd_file}" "${lmbda}" "${temp}"; then
        echo "== ${method} day ${day}: infer skip lmbda=${lmbda} temp=${temp}"
        continue
      fi
      local gpu="${gpus[$gpu_idx]}"
      (
        cd "${SCRIPT_DIR}"
        bash ./pipe.sh --action infer_one ${command_common} --gpu "${gpu}" --stop "" --lmbda "${lmbda}" --knn-temp "${temp}"
      ) &
      jobs+=("$!")
      gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))
      if [[ "${#jobs[@]}" -ge "${#gpus[@]}" ]]; then
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
  if [[ "${status}" -ne 0 ]]; then
    echo "One or more infer workers failed for ${method} day ${day}; retrying missing metric pairs serially" >&2
    status=0
    for lmbda in $(seq 0.4 0.05 0.6); do
      for temp in $(seq 8.5 0.5 15.5); do
        if metric_pair_done "${ppl_file}" "${pwd_file}" "${lmbda}" "${temp}"; then
          continue
        fi
        if ! run_pipe --action infer_one ${command_common} --gpu "${gpus[0]}" --stop "" --lmbda "${lmbda}" --knn-temp "${temp}"; then
          status=1
        fi
      done
    done
    if [[ "${status}" -ne 0 ]]; then
      echo "One or more serial infer retries failed for ${method} day ${day}" >&2
      exit "${status}"
    fi
  fi
}

IFS=',' read -r -a method_list <<< "${METHODS}"
for method in "${method_list[@]}"; do
  method="$(echo "${method}" | tr '[:upper:]' '[:lower:]' | xargs)"
  read -r pt pr <<< "$(method_params "${method}")"
  for day in $(seq "${BEGIN_DAY}" 1 "${END_DAY}"); do
    param="${pt},${pr}"
    prefix="${CKPT_NAME}_${day}__fp16_p_${param}.txt"
    if [[ "${EVAL_EACH_DAY}" -eq 1 ]] && metric_grid_done "${ROOT}/output/debug/ppl/${prefix}" "${ROOT}/output/debug/pwd/${prefix}"; then
      echo "== ${method} day ${day}: complete metric grid; skip"
      continue
    fi
    if [[ "${EVAL_ONLY}" -eq 0 ]]; then
      run_day "${method}" "${day}" "${pt}" "${pr}"
    fi
    if [[ "${EVAL}" -eq 1 && "${EVAL_EACH_DAY}" -eq 1 ]]; then
      run_eval "${method}" "${day}" "${pt}" "${pr}"
    fi
  done
  if [[ "${EVAL}" -eq 1 && "${EVAL_EACH_DAY}" -eq 0 ]]; then
    run_eval "${method}" "${END_DAY}" "${pt}" "${pr}"
  fi
done

if [[ "${EVAL}" -eq 1 ]]; then
  "${PYTHON_BIN}" "${ROOT}/monim/summarize_results.py" --begin "${BEGIN_DAY}" --end "${END_DAY}" --methods "${METHODS}"
fi
