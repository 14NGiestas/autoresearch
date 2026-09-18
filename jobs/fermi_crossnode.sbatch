#!/bin/bash
#SBATCH -J crossnode
#SBATCH -c 8
#SBATCH -t 00:60:00
#SBATCH -o /home/pauli/autoresearch/logs/crossnode_%j.log
# Cross-node de verdade: fermi (rank 0) + halfbeast (rank 1), 1 rank por no.
# (a) hostname nos 2 nos, (b) invariante com dados iguais vs 1 rank local,
# (c) throughput 1 rank local x 2 ranks cross-node em tau=1 e tau=100.
# Teto 6G (mpi/guard.sh): train_ddp cross-node roda o modelo em cada no.
# SEGURANCA: -envlist explicito em todo mpirun (nunca ambiente herdado).
set -u
cd /home/pauli/autoresearch
# Portao MEMORIA-CONSCIENTE (nao mais "zero jobs rodando"): o que causou o
# global_oom de 07:18 foi um mpirun SEM teto de cgroup em paralelo; com o escopo
# do guard.sh o meu lado morre sozinho se estourar. Entao a regra agora e':
# re-enfileira somente se a memoria disponivel nao comportar os dois jobs.
# MIN_FREE_MB=8000 por default (o job pede 6G de teto; a fermi tem 15,4 GB).
OTHERS=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" 2>/dev/null | grep -v "^${SLURM_JOB_ID:-0} " || true)
AVAIL_MB=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
MIN_FREE_MB=${MIN_FREE_MB:-8000}
echo "portao: outros jobs RUNNING=[$(echo "$OTHERS" | tr '\n' ';')]  MemAvailable=${AVAIL_MB} MB (minimo p/ co-rodar: ${MIN_FREE_MB} MB)"
if [ -n "$OTHERS" ] && [ "${AVAIL_MB:-0}" -lt "$MIN_FREE_MB" ]; then
  echo "portao: memoria insuficiente p/ co-rodar -- re-enfileirando:"
  echo "$OTHERS" | sed 's/^/  /'
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
chmod +x mpi/guard.sh mpi/crossnode_stage.sh mpi/tests_crossnode.sh
exec bash mpi/guard.sh 6G nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich -c \
  bash -c 'bash mpi/crossnode_stage.sh && bash mpi/tests_crossnode.sh'
