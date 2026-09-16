#!/bin/bash
#SBATCH -J compose
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/compose_%j.log
# Taxonomia da composicao linear: disj/order/seed/lr/anneal (ver scripts/compose_grid.py).
set -u
cd /home/pauli/autoresearch
echo "compose job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
.venv-numpy/bin/python3 scripts/compose_grid.py --fams "${1:-disj,order,seed,lr,anneal}" --out "/tmp/compose${2:-}"
echo "fim compose $(date '+%F %H:%M:%S')"
