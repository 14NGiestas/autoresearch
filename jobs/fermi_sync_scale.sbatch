#!/bin/bash
#SBATCH -J syncscale
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o /home/pauli/autoresearch/logs/syncscale_%j.log
# Com a config vencedora (ZERA os momentos no sync + outer Nesterov lr=0.7),
# duas perguntas: (1) tau menor ajuda mais? (2) o ganho escala com K?
# Ja temos com essa config: tau=488 -> 2.55000 e tau=122 -> 2.33183 (ambos K=4,
# 1952 passos/worker). A maquina unica no mesmo wall-clock da 2.44475.
# IDEAL (referencia honesta): 1 maquina com K x 1952 passos nos dados completos:
#   K=4 -> 7808 passos -> 2.12000 (medido)   |   K=8 -> 15616 passos -> 2.02520 (medido)
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "syncscale job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
echo "--- (1) tau=61 (32 syncs), K=4, config vencedora"
$PY scripts/fedavg_rounds.py --k 4 --tau 61 --rounds 32 --out /tmp/ss/t61 \
    --reset-moments --outer nesterov --outer-lr 0.7 --outer-mom 0.9
echo "--- (2) K=8, tau=122 (16 syncs), config vencedora"
$PY scripts/fedavg_rounds.py --k 8 --tau 122 --rounds 16 --out /tmp/ss/k8t122 \
    --reset-moments --outer nesterov --outer-lr 0.7 --outer-mom 0.9
echo "fim syncscale $(date '+%F %H:%M:%S')"
