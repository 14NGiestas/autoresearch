#!/bin/bash
#SBATCH -J rot
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o /home/pauli/autoresearch/logs/rot_%j.log
# DATA-PARALLEL de verdade (o braco "parallel" do job 97 estava errado: phase
# pequeno = K janelas quase iguais, medindo cobertura, nao regime). Aqui os K
# workers pegam K blocos CONSECUTIVOS e DISJUNTOS por rodada, e os blocos avancam
# K*tau por rodada: ao longo do run cada worker ve o pool inteiro (ninguem fica
# preso a uma fatia). Se sincronizar ajuda em algum arranjo, e aqui.
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "rot job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
$PY scripts/fedavg_rounds.py --regime rotating --k 4 --tau 1952 --rounds 1  --out /tmp/rot/t1952
$PY scripts/fedavg_rounds.py --regime rotating --k 4 --tau 488  --rounds 4  --out /tmp/rot/t488
$PY scripts/fedavg_rounds.py --regime rotating --k 4 --tau 122  --rounds 16 --out /tmp/rot/t122
$PY scripts/fedavg_rounds.py --regime rotating --k 4 --tau 48   --rounds 40 --out /tmp/rot/t48
echo "fim rot $(date '+%F %H:%M:%S')"
