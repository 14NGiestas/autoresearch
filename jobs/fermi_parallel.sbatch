#!/bin/bash
#SBATCH -J parallel
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/parallel_%j.log
# O regime que a meta de 6 meses realmente habita: DATA-PARALLEL (mesma
# distribuicao, fases diferentes), nao federado (fatias disjuntas). Aqui a media
# e a operacao CORRETA (os K gradientes estimam o MESMO gradiente), entao o
# sync deve AJUDAR em vez de atrapalhar. Comparacao: mesma K, mesmo compute
# (7808 passos), mesmo wall-clock (1952 por worker) dos bracos federados.
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
echo "parallel job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
$PY scripts/fedavg_rounds.py --regime parallel --k 4 --tau 1952 --rounds 1  --out /tmp/par/t1952
$PY scripts/fedavg_rounds.py --regime parallel --k 4 --tau 488  --rounds 4  --out /tmp/par/t488
$PY scripts/fedavg_rounds.py --regime parallel --k 4 --tau 122  --rounds 16 --out /tmp/par/t122
echo "fim parallel $(date '+%F %H:%M:%S')"
