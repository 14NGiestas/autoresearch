#!/bin/bash
#SBATCH -J sync
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -o /home/pauli/autoresearch/logs/sync_%j.log
# FEDERADO (fatias disjuntas) vs DATA-PARALLEL (mesma distribuicao, fase), com
# start_row AVANCANDO por rodada (o bug anterior replayava as mesmas linhas em
# todas as rodadas, o que era um terceiro experimento). Controle tau=infinito
# (1 rodada) e barato e entra como baseline do proprio driver.
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "sync job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
# data-parallel: mesma distribuicao, fases, 1/4/16 syncs
$PY scripts/fedavg_rounds.py --regime parallel  --k 4 --tau 1952 --rounds 1  --out /tmp/par2/t1952
$PY scripts/fedavg_rounds.py --regime parallel  --k 4 --tau 488  --rounds 4  --out /tmp/par2/t488
$PY scripts/fedavg_rounds.py --regime parallel  --k 4 --tau 122  --rounds 16 --out /tmp/par2/t122
# federado corrigido (fatias disjuntas, dados avancando)
$PY scripts/fedavg_rounds.py --regime federated --k 4 --tau 488  --rounds 4  --out /tmp/fed2/t488
$PY scripts/fedavg_rounds.py --regime federated --k 4 --tau 122  --rounds 16 --out /tmp/fed2/t122
echo "fim sync $(date '+%F %H:%M:%S')"
