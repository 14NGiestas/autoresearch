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
chmod +x mpi/guard.sh mpi/crossnode_stage.sh mpi/crossnode_debug.sh
exec bash mpi/guard.sh 6G nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich -c \
  bash -c 'bash mpi/crossnode_stage.sh && bash mpi/crossnode_debug.sh'
