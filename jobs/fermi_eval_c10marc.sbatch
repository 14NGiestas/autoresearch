#!/bin/bash
#SBATCH -J c10marc
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/c10marc_%j.log
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d216_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
echo "c10marc job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/w_c10m/step_195312 --rows /tmp/bench/arc_rows.npy --attn blas --batch 2 \
  > /tmp/ev_arc_c10m.txt 2>&1
echo "fim c10marc $(date '+%F %H:%M:%S')"
