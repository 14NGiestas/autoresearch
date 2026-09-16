#!/bin/bash
#SBATCH -J evalc10m
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/evalc10m_%j.log
# Val REAL do c10m (treino nao tinha linhas de val no npy).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d216_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
echo "evalc10m job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/w_c10m/step_195312 --rows /tmp/mix200/val_mix200.npy --attn blas --batch 4 \
  > /tmp/ev_c10m.txt 2>&1
echo "fim evalc10m $(date '+%F %H:%M:%S')"
