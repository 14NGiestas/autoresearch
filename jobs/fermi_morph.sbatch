#!/bin/bash
#SBATCH -J morph01
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /home/pauli/autoresearch/logs/morph01_%j.log
# TOK-LEX segundo braco: MorphBPE (merges nunca cruzam fronteira morfemica).
# Mesmo texto/init/LR/val do f0. Prediz: H2 desce (composicoes validas).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
echo "morph p=bound rows=11550 job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/init3m --rows /tmp/mix/rows_f_morph.npy --out /tmp/mix/w_f_morph \
  --nsteps 11550 --lr 0.0006 --ntrain 11550 --nval 5 --val_every 1000 \
  --save_every 4000 --keep_last 2 --trn_probe 200 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
echo "fim morph01 $(date '+%H:%M:%S')"
