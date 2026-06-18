# MoNIM Reproduction

This directory contains the open-source reproduction layer for the default
experiments:

- `launch_cl_1_30.sh`: runs FullMem datastore/index construction, evaluates
  FullMem, runs MoNIM, and finalizes result CSVs.
- `launch_fullmem_1_30.sh`: evaluates FullMem using existing FullMem
  datastores and indexes.
- `run_cl_1_30.sh`: builds continual kNN-LM artifacts and runs MoNIM.
- `eval_fullmem_1_30.sh`: runs the FullMem metric
  grid over days 1-30.
- `finalize_results.sh`: validates artifacts and writes default result tables.
- `common.sh`: shared shell environment setup for launch and evaluation scripts.
- `artifact_status.py`: artifact and metric-grid status checks used by shell
  scripts.
- `result_utils.py`: shared method definitions, metric parsing, and CSV writers.
- `default_results.py`, `summarize_results.py`, `check_metric_grid.py`,
  `audit_artifacts.py`: result collection and validation utilities.
- `adaptive_retrieval/`: MoNIM feature extraction and adapter training code.
- `moe_modules/`: neural modules used by the MoNIM adapter.

Expected repository layout:

```text
knnlm/
monim/
datasets/
checkpoints/
```

Runtime artifacts are intentionally not part of this source tree. Scripts write
datastores, indexes, logs, and result files to `storage/`, `output/`, and
`monim/final/` at the repository root.

Default reproduction:

```bash
python -m venv .venv-uv
. .venv-uv/bin/activate
pip install -r monim/requirements.txt
pip install -e knnlm
GPUS=0,1,2,3 bash monim/launch_cl_1_30.sh
```
