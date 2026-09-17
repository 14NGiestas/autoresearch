#!/bin/bash
#SBATCH -J rescale
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/rescale_%j.log
# Mecanismo do imposto de composicao: a media encolhe (s~0.9). Reescalar a media
# recupera o bpb? Varredura de escala nos merges ja existentes (nenhum treino).
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
echo "rescale job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
for M in disj2 disj4; do
  for S in 0.85 0.90 0.95 1.00 1.05 1.10 1.15; do
    TAG="${M}_x${S}"
    D=/tmp/compose/$TAG
    [ -d "$D" ] || $PY scripts/rescale_ckpt.py /tmp/compose/${M}_merge "$D" --scale $S >/dev/null
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
      --weights "$D" --rows /tmp/mix/rows_holdout.npy --attn blas --batch 16 \
      > /tmp/compose/ev_$TAG.txt 2>&1
    echo "  $TAG ok"
  done
done
echo "fim rescale $(date '+%F %H:%M:%S')"
