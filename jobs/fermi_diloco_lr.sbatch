#!/bin/bash
#SBATCH -J dlsweep
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o /home/pauli/autoresearch/logs/dlsweep_%j.log
# O DiLoCo com lr_outer=0.7 (valor do paper) deu 2.55000 contra 2.54089 do
# federado puro: NEUTRO. O lr_outer e' O botao do DiLoCo e 0.7 e' o valor da
# escala deles. Aqui: sweep em tau=488 para ver se existe faixa util -- se
# existir, vale testar tau=122 (onde o imposto e' maior e ha mais a consertar).
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "dlsweep job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
for LR in 0.1 0.3 1.5; do
  echo "--- lr_outer=$LR"
  $PY scripts/fedavg_rounds.py --k 4 --tau 488 --rounds 4 --out /tmp/dlsweep/$LR \
      --outer nesterov --outer-lr $LR --outer-mom 0.9
done
echo "fim dlsweep $(date '+%F %H:%M:%S')"
