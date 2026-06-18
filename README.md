# MoNIM

Code and resources for **[Learn to Memorize: Scalable Continual Learning in Semiparametric Models with Mixture-of-Neighbors Induction Memory](https://aclanthology.org/2025.acl-long.1385/)**, ACL 2025.

MoNIM treats non-parametric memory in semiparametric language models as a learnable Mixture-of-Neighbors Induction Memory. This repository contains the kNN-LM runtime, FullMem baseline, MoNIM adapter code, and scripts for reproducing the 1-30 day continual-learning experiments.

## 1. Environment

Clone the repository and enter the project root.

```bash
cd MoNIM
```

Create the Python environment. The scripts default to `.venv-uv/bin/python`, so
the environment is created at `./.venv-uv`.

```bash
conda create -y -p ./.venv-uv python=3.8
conda activate ./.venv-uv
conda install -y -c pytorch -c nvidia faiss-gpu
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r monim/requirements.txt
python -m pip install -e knnlm
```

The default reproduction uses CUDA GPUs and FAISS GPU. If you use a different
environment path, set `KNNLM_VENV=/path/to/env` before running the scripts.

```bash
export KNNLM_VENV=/path/to/env
```

## 2. Models

The released reproduction checkpoint is available with the data at:

https://huggingface.co/datasets/viniferagy/MoNIM

After downloading the Hugging Face repository in Section 3, place or link the
checkpoint under `checkpoints/`.

```text
checkpoints/
  gpt2-small/
    checkpoint_best.pt
```

The default scripts use the model key:

```text
gpt2-small -> checkpoints/gpt2-small/checkpoint_best.pt
```

## 3. Data

The released 1-30 day reproduction data and GPT-2 small checkpoint are available at:

https://huggingface.co/datasets/viniferagy/MoNIM

Download the repository, then expose the data and checkpoint paths expected by
the scripts.

```bash
git lfs install
git clone https://huggingface.co/datasets/viniferagy/MoNIM hf_assets
ln -sfn hf_assets datasets
ln -sfn hf_assets/checkpoints checkpoints
```

Expected data layout:

```text
checkpoints/
  gpt2-small/
    checkpoint_best.pt
datasets/
  dict.txt
  encoder.json
  vocab.bpe
  daily/
    1/
      train.txt
      valid.txt
      test.txt
      train.bpe
      valid.bpe
      test.bpe
      bin/
      net/
    ...
    30/
```

Runtime artifacts are written outside the released dataset:

```text
storage/
output/
datasets/daily/<day>/net/total/gpt2-small/-1.5_1.0/
monim/logs/
monim/final/
```

## 4. Run Pipeline

Run the full day 1-30 reproduction with four GPUs.

```bash
REPRO_GPUS=0,1,2,3 bash monim/launch_cl_1_30.sh
```

This pipeline:

```text
1. builds FullMem datastores and FAISS indexes;
2. evaluates FullMem;
3. runs MoNIM and trains the MoNIM adapter from the released data;
4. evaluates MoNIM on every day;
5. writes the final CSV files.
```

Run the stages manually when resuming or debugging.

```bash
bash monim/run_cl_1_30.sh \
  --methods fullmem \
  --begin 1 \
  --end 30 \
  --gpus 0,1,2,3 \
  --no-eval

GPUS=0,1,2,3 bash monim/eval_fullmem_1_30.sh

bash monim/run_cl_1_30.sh \
  --methods monim \
  --begin 1 \
  --end 30 \
  --gpus 0,1,2,3 \
  --eval-each-day

bash monim/finalize_results.sh
```

If adapter checkpoints already exist under `datasets/daily/<day>/net/total/gpt2-small/-1.5_1.0/checkpoint/`, MoNIM can skip adapter training.

```bash
bash monim/run_cl_1_30.sh \
  --methods monim \
  --begin 1 \
  --end 30 \
  --gpus 0,1,2,3 \
  --reuse-net \
  --eval-each-day
```

## 5. Outputs

Logs are written to:

```text
monim/logs/
```

KNN datastore and FAISS index artifacts are written to:

```text
storage/gpt2-small/daily/dstore/
storage/gpt2-small/daily/knn/
output/size.json
```

Raw metric grids are written to:

```text
output/debug/ppl/
output/debug/pwd/
monim/final/fullmem_output/debug/ppl/
monim/final/fullmem_output/debug/pwd/
```

Final evaluation outputs are written to:

```text
monim/final/results_1_30.csv
monim/final/metric_grid_1_30.csv
monim/final/artifact_audit_1_30.csv
monim/final/fullmem_results_1_30.csv
monim/final/fullmem_metric_grid_1_30.csv
```

Regenerate the final CSVs from completed metrics.

```bash
bash monim/finalize_results.sh
```

Inspect MoNIM results only.

```bash
python monim/summarize_results.py \
  --begin 1 \
  --end 30 \
  --methods monim \
  --strict
```

Inspect metric-grid completeness.

```bash
python monim/check_metric_grid.py \
  --begin 1 \
  --end 30 \
  --methods fullmem,monim \
  --strict
```

## 6. Citation

```bibtex
@inproceedings{peng-etal-2025-learn,
  title = "Learn to Memorize: Scalable Continual Learning in Semiparametric Models with Mixture-of-Neighbors Induction Memory",
  author = "Peng, Guangyue  and
    Ge, Tao  and
    Luo, Wen  and
    Li, Wei  and
    Wang, Houfeng",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2025",
  address = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.acl-long.1385/",
  doi = "10.18653/v1/2025.acl-long.1385",
  pages = "28517--28531",
}
```
