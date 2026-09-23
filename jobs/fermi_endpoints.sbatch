#!/bin/bash
#SBATCH -J endpoints
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /home/pauli/autoresearch/logs/endpoints_%j.log
# Extremos do eixo f0-f100 nas MESMAS rows do barrier (fecha o perfil).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE
for A in f0 f100; do
  "$BIN" --weights /tmp/mix/w_$A/best --rows /tmp/mix/rows_f0.npy --attn blas --batch 8 > /tmp/ev_bar_end_$A.txt 2>&1
  echo "$A done $(date '+%F %H:%M:%S')"
done
echo "fim endpoints $(date '+%F %H:%M:%S')"
