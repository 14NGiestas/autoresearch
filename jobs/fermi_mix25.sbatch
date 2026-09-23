#!/bin/bash
#SBATCH -J mix25
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/mix25_%j.log
# Mix 25M: 64M tokens (~2.5 tok/param), f=25, T=1024.
# Literario 48M (prosa+wikisource por estoque) + wiki 16M.
set -u
cd /home/pauli/autoresearch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-numpy/bin/python3 scripts/build_mix.py \
  --prose /tmp/prose/prose_bpe_all.txt --prose-ntrain 59385 \
  --prose2 /tmp/ptwikisource/ws_bpe_all.txt \
  --prose-tokens 48e6 --wiki-tokens 16e6 --block 32768 --T 1024 \
  --out /tmp/mix/rows_m25.txt --val-out /tmp/mix/val_m25.txt \
  --manifest /tmp/mix/manifest_m25.json
