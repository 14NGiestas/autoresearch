#!/bin/bash
#SBATCH -J lrtax
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /home/pauli/autoresearch/logs/lrtax_%j.log
# Fecha a lei do imposto de composicao em funcao do LR: o grid de composicao so
# avaliou o MERGE dos bracos de lr (nao os shards), entao o imposto nao pode ser
# calculado. Aqui so avaliamos os shards que ja estao em disco (nenhum treino).
set -u
cd /home/pauli/autoresearch
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
echo "lrtax job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
for arm in lr00002 lr64447; do
  for k in 0 1 2 3; do
    D=$(ls -d /tmp/compose/${arm}_s$k/step_* 2>/dev/null | head -1)
    [ -z "$D" ] && { echo "  $arm s$k ausente"; continue; }
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
      --weights "$D" --rows /tmp/mix/rows_holdout.npy --attn blas --batch 16 \
      > /tmp/compose/ev_${arm}_s$k.txt 2>&1
    echo "  $arm s$k ok"
  done
done
echo "fim lrtax $(date '+%F %H:%M:%S')"
