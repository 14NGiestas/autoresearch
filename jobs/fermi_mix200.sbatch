#!/bin/bash
#SBATCH -J mix200
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/mix200_%j.log
# Mix-200M (Chinchilla-10M): prosa 60M + wiki 100M + SO 25M + instruct 15M.
set -u
cd /home/pauli/autoresearch
mkdir -p /tmp/mix200
.venv-numpy/bin/python3 scripts/tokenize_budget.py --fmt so \
  --src /tmp/bench/so_pairs.jsonl --budget-tokens 25e6 --out /tmp/mix200/so_bpe.txt
.venv-numpy/bin/python3 scripts/tokenize_budget.py --fmt instr \
  --src "/tmp/bench/instr_*.parquet" --budget-tokens 15e6 --out /tmp/mix200/instr_bpe.txt
cat /tmp/mix200/so_bpe.txt /tmp/mix200/instr_bpe.txt > /tmp/mix200/other_bpe.txt
.venv-numpy/bin/python3 scripts/build_mix.py \
  --prose /tmp/prose/prose_bpe_all.txt --prose-ntrain 59385 \
  --prose2 /tmp/mix200/other_bpe.txt --prose-tokens 100e6 --wiki-tokens 100e6 \
  --block 32768 --T 1024 --out /tmp/mix200/rows_mix200.txt \
  --val-out /tmp/mix200/val_mix200.txt --manifest /tmp/mix200/manifest.json
echo "fim mix200 $(date '+%F %H:%M:%S')"
