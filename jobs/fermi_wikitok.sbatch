#!/bin/bash
#SBATCH -J wikitok
#SBATCH --array=0-7
#SBATCH -c 1
#SBATCH -t 01:00:00
#SBATCH -o /home/pauli/autoresearch/logs/wikitok_%A_%a.log
# Tokeniza wiki_filt.jsonl em BPE (8 chunks independentes, ~2-4 min).
set -u
cd /home/pauli/autoresearch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
i=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
.venv-numpy/bin/python3 scripts/tokenize_wiki_bpe.py \
  --src /tmp/ptwiki/chunk_$i.jsonl --out /tmp/ptwiki/wiki_bpe_$i.txt
