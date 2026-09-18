#!/bin/bash
#SBATCH -J crossdbg
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o /home/pauli/autoresearch/logs/crossdbg_%j.log
# Diagnostico do lancamento remoto do hydra (fermi + halfbeast) e staging em
# /tmp/ddp nos dois nos. So' testes triviais (hostname) + probe de rede: teto de
# 4G (a regra diz 4G para MPI puro; o train_ddp cross-node vai em outro job com 6G).
# SEGURANCA: todo mpirun usa -envlist explicito (nunca ambiente herdado).
set -u
cd /home/pauli/autoresearch
OTHERS=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" 2>/dev/null | grep -v "^${SLURM_JOB_ID:-0} " || true)
if [ -n "$OTHERS" ]; then
  echo "portao: outro(s) job(s) RUNNING -- re-enfileirando:"; echo "$OTHERS" | sed 's/^/  /'
  scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
chmod +x mpi/guard.sh mpi/crossnode_stage.sh mpi/crossnode_debug.sh
exec bash mpi/guard.sh 6G nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich -c \
  bash -c 'bash mpi/crossnode_stage.sh && bash mpi/crossnode_debug.sh'
