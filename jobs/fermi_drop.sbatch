#!/bin/bash
#SBATCH -J drop01
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /home/pauli/autoresearch/logs/drop01_%j.log
# TOK-LEX intervencao: BPE-dropout p=0.1 no MESMO texto do f0 (3M, 1 epoca).
# Comparavel direto vs f0 (mesmo init, LR, val). Prediz: H2 desce, fert sobe.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
echo "dropout p=0.1 rows=8784 job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/init3m --rows /tmp/mix/rows_f_drop.npy --out /tmp/mix/w_f_drop \
  --nsteps 8784 --lr 0.0006 --ntrain 8784 --nval 5 --val_every 1000 \
  --save_every 4000 --keep_last 2 --trn_probe 200 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
echo "fim drop01 $(date '+%H:%M:%S')"
