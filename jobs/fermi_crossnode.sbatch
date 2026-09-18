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
OTHERS=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" 2>/dev/null | grep -v "^${SLURM_JOB_ID:-0} " || true)
if [ -n "$OTHERS" ]; then
  echo "portao: outro(s) job(s) RUNNING -- re-enfileirando:"; echo "$OTHERS" | sed 's/^/  /'
  scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
chmod +x mpi/guard.sh mpi/crossnode_stage.sh mpi/tests_crossnode.sh
exec bash mpi/guard.sh 6G nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich -c \
  bash -c 'bash mpi/crossnode_stage.sh && bash mpi/tests_crossnode.sh'
