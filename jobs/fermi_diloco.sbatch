#!/bin/bash
#SBATCH -J diloco
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/diloco_%j.log
# O ULTIMO botao nao testado: otimizador EXTERNO (DiLoCo). Sem ele, media de
# pesos com tau finito PIORA com mais sync (2.545 com 4 syncs, 2.690 com 16,
# contra 2.504 do one-shot). Com ele, o outer acumula Nesterov sobre o
# pseudo-gradiente e deve recuperar a qualidade -- e a literatura diz que escala.
# Mesmo K/compute/wall-clock dos bracos federados: comparacao direta.
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "diloco job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
for TAU in 488 122; do
  R=$(( 7808 / (4*TAU) ))
  $PY scripts/fedavg_rounds.py --k 4 --tau $TAU --rounds $R --out /tmp/dl/$TAU \
      --outer nesterov --outer-lr 0.7 --outer-mom 0.9
done
echo "fim diloco $(date '+%F %H:%M:%S')"
