#!/bin/bash
#SBATCH -J synclr
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /home/pauli/autoresearch/logs/synclr_%j.log
# tau x lr_outer: o achatamento do tau=61 (2.32519) contra tau=122 (2.33183) e'
# real ou o lr=0.7 (ajustado em tau=488) esta' errado para tau pequeno?
# Bracket: lr_outer 0.4 e 1.0 em tau=61 (32 rodadas) e em tau=244 (8 rodadas).
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "synclr job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
for lr in 0.4 1.0; do
  echo "=== tau=61 (32 syncs), lr_outer=$lr"
  $PY scripts/fedavg_rounds.py --k 4 --tau 61 --rounds 32 --out /tmp/ss/t61_lr$lr \
      --reset-moments --outer nesterov --outer-lr $lr --outer-mom 0.9
done
for lr in 0.4 0.7 1.0; do
  echo "=== tau=244 (8 syncs), lr_outer=$lr"
  $PY scripts/fedavg_rounds.py --k 4 --tau 244 --rounds 8 --out /tmp/ss/t244_lr$lr \
      --reset-moments --outer nesterov --outer-lr $lr --outer-mom 0.9
done
echo "fim synclr $(date '+%F %H:%M:%S')"
