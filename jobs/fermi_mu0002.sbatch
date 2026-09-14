#!/bin/bash
#SBATCH -J mu0002
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/mu0002_%j.log
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
echo "mu0002 lr-muon=0.002 job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/mix/w_f_mu0002 \
  --nsteps 2000 --lr 0.0006 --opt muon --muon-lr 0.002 --ntrain 7812 --nval 5 \
  --val_every 500 --save_every 2000 --keep_last 1 --trn_probe 200 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
echo "fim mu0002 $(date '+%H:%M:%S')"
