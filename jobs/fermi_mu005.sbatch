#!/bin/bash
#SBATCH -J mu005
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/mu005_%j.log
# Sweep lr-muon (sonda 2000 steps): mesmo rows/init do f0.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
echo "mu005 lr-muon=0.05 job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/mix/w_f_mu005 \
  --nsteps 2000 --lr 0.0006 --opt muon --muon-lr 0.05 --ntrain 7812 --nval 5 \
  --val_every 500 --save_every 2000 --keep_last 1 --trn_probe 200 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
echo "fim mu005 $(date '+%H:%M:%S')"
