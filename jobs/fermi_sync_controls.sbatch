#!/bin/bash
#SBATCH -J synccontrol
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o /home/pauli/autoresearch/logs/synccontrol_%j.log
# CONTROLES do 2x2 (tau=122, 16 rodadas, 1952 passos/worker).
# Pergunta 1: o ganho do outer (-0.270) e' "conserto de sincronizacao" ou so'
#   "Nesterov externo e' otimizador melhor"? -> K=1 com e sem outer, em 3 pools.
#   (a literatura diz que single-worker DiLoCo ja' bate AdamW puro: se for isso
#    aqui, o numero e' de otimizador, nao de composicao)
# Pergunta 2: o que envenena o sync e' MEDIAR os momentos ou CARREGAR momentos?
#   -> K=4 com momentos POR WORKER (sem media), com e sem outer.
# Pools: f0 (o de sempre) + perm1/perm2 = mesma distribuicao, ordem sorteada
#   (nao ha flag de seed: a unica aleatoriedade e' o sorteio dos dados).
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "synccontrol job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
for pool in f0 perm1 perm2; do
  echo "=== pool=$pool : referencia (run unico de 1952 passos = 1 rodada)"
  $PY scripts/fedavg_rounds.py --k 1 --tau 1952 --rounds 1 --out /tmp/ctrl/plain_$pool \
      --rows /tmp/mix/rows_$pool.npy
  echo "=== pool=$pool : K=1, tau=122, 16 rodadas, SEM outer (so' reiniciar o Adam)"
  $PY scripts/fedavg_rounds.py --k 1 --tau 122 --rounds 16 --out /tmp/ctrl/k1_$pool \
      --rows /tmp/mix/rows_$pool.npy
  echo "=== pool=$pool : K=1, tau=122, 16 rodadas, outer 0.7 (DiLoCo de 1 worker)"
  $PY scripts/fedavg_rounds.py --k 1 --tau 122 --rounds 16 --out /tmp/ctrl/k1outer_$pool \
      --rows /tmp/mix/rows_$pool.npy --outer nesterov --outer-lr 0.7 --outer-mom 0.9
done
echo "=== K=4, tau=122: momentos POR WORKER (sem media), sem outer"
$PY scripts/fedavg_rounds.py --k 4 --tau 122 --rounds 16 --out /tmp/ctrl/pw_k4 \
    --reset-moments --carry-per-worker
echo "=== K=4, tau=122: momentos POR WORKER + outer 0.7"
$PY scripts/fedavg_rounds.py --k 4 --tau 122 --rounds 16 --out /tmp/ctrl/pw_k4outer \
    --reset-moments --carry-per-worker --outer nesterov --outer-lr 0.7 --outer-mom 0.9
echo "fim synccontrol $(date '+%F %H:%M:%S')"
