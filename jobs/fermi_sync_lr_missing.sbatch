#!/bin/bash
#SBATCH -J synclrmiss
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /home/pauli/autoresearch/logs/synclrmiss_%j.log
# Completa a tabela tau x lr_outer: o job 138 (limite 4h) vai ser cortado antes
# dos dois ultimos arms (tau=244 com lr_outer 0.7 e 1.0). Aqui so' eles.
#
# Por que a tabela importa: o tau=61 (32 syncs) EMPATA com tau=122 no final
# (2.32519 vs 2.33183), mas o lr=0.7 foi ajustado em tau=488. Se tau menor nao
# melhora por causa de lr errado, o "saturacao" e' artefato. Com tau=244 em tres
# lr (0.4 ja' sai no 138; 0.7 e 1.0 aqui) da' para ver se o vale de lr se move
# com tau.
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "synclrmiss job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
for lr in 0.7 1.0; do
  echo "=== tau=244 (8 syncs), lr_outer=$lr"
  $PY scripts/fedavg_rounds.py --k 4 --tau 244 --rounds 8 --out /tmp/ss/t244_lr$lr \
      --reset-moments --outer nesterov --outer-lr $lr --outer-mom 0.9
done
echo "fim synclrmiss $(date '+%F %H:%M:%S')"
