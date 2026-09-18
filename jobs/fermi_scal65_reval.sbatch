#!/bin/bash
#SBATCH -J reval
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/reval_%j.log
# A curva do pool limpo (32 pontos) ficou com ruido de +-0,04 bpb entre
# checkpoints vizinhos -- o holdout de 300 linhas nao resolve diferencas de 0,01.
# Aqui so RE-AVALIAMOS os 32 checkpoints que ja existem num holdout de 1000
# linhas (linhas 63900-64900, virgens). Zero treino: o gargalo e a medicao.
set -u
cd /home/pauli/autoresearch
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
mkdir -p /tmp/scal65_1k
echo "reval job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
n=0
for d in /tmp/scal65/step_* /tmp/scal65_tail/step_*; do
  s=$(basename $d | sed 's/step_//')
  [ -f "$d/model.safetensors" ] || continue
  [ -s "/tmp/scal65_1k/ev_$s.txt" ] && continue   # idempotente: pula o que ja' foi avaliado
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
    --weights "$d" --rows /tmp/prose65/rows_val1k.npy --attn blas --batch 16 \
    > /tmp/scal65_1k/ev_$s.txt 2>&1
  n=$((n+1))
done
echo "  $n checkpoints reavaliados"
echo "fim reval $(date '+%F %H:%M:%S')"
