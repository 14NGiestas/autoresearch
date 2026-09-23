#!/bin/bash
#SBATCH -J c10m
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH -o /home/pauli/autoresearch/logs/c10m_%j.log
# Chinchilla-10M: d216, 200M tokens distintos (~20 tok/param), mix-200M.
set -u
cd /home/pauli/autoresearch
E=$(ls -t src/build/arch_d216_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE
"$E" --weights /tmp/mix200/init10m --rows /tmp/mix200/rows_mix200.npy --out /tmp/w_c10m \
  --nsteps 195312 --lr 0.0003 --start_row 0 --ntrain 195312 --nval 5 \
  --val_every 2000 --save_every 8000 --keep_last 2 --trn_probe 200 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
echo "fim c10m $(date '+%F %H:%M:%S')"
