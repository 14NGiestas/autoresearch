#!/bin/bash
#SBATCH -J barrier
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /home/pauli/autoresearch/logs/barrier_%j.log
# Barreira f0->f100: bpb em prosa ao longo da interpolacao (pais + 3 alphas).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE
for A in 0.25 0.5 0.75; do
  T=$(echo $A | tr -d .); OUT=/tmp/w_barrier_$T
  .venv-numpy/bin/python3 scripts/merge_checkpoints.py /tmp/mix/w_f0/best /tmp/mix/w_f100/best $OUT --alpha $A 2>&1 | tail -1
  "$BIN" --weights $OUT --rows /tmp/mix/rows_f0.npy --attn blas --batch 8 > /tmp/ev_bar_$T.txt 2>&1
  echo "alpha=$A done $(date '+%F %H:%M:%S')"
done
echo "fim barrier $(date '+%F %H:%M:%S')"
