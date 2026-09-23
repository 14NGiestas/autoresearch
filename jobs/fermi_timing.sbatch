#!/bin/bash
#SBATCH -J timing3m
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o /home/pauli/autoresearch/logs/timing3m_%j.log
# Throughput do d96 (3M) para dimensionar o grid de composicao.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
echo "timing job=${SLURM_JOB_ID} host=$(hostname)"
rm -rf /tmp/timing
S=$(date +%s.%N)
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/timing \
  --nsteps 400 --lr 0.0006 --ntrain 7812 --nval 1 --val_every 100000 --attn blas \
  --save_every 100000 --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt" \
  > /tmp/timing.log 2>&1
E=$(date +%s.%N)
echo "400 steps em $(echo "$E-$S"|bc)s -> tok/s=$(echo "scale=0; 400*1024/($E-$S)"|bc)"
echo "fim timing"
