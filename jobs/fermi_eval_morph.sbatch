#!/bin/bash
#SBATCH -J evalmorph
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /home/pauli/autoresearch/logs/evalmorph_%j.log
# bpb JUSTO do morph: val codificada em morph (mesmo texto, outra segmentacao).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
echo "evalmorph job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/w_f_morph/best --rows /tmp/mix/val_morph.npy --attn blas --batch 8 \
  > /tmp/ev_morphfair.txt 2>&1
echo "fim evalmorph $(date '+%F %H:%M:%S')"
