#!/bin/bash
#SBATCH -J evalties
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/evalties_%j.log
# Eval full-file da sopa TIES (val-only calculado depois via bpb_from_eval).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
echo "evalties job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/w_ovsoup_ties --rows /tmp/mix/rows_f0.npy --attn blas --batch 8 \
  > /tmp/ev_soupties.txt 2>&1
echo "fim evalties $(date '+%F %H:%M:%S')"
