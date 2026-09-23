#!/bin/bash
#SBATCH -J mpiddp
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /home/pauli/autoresearch/logs/mpiddp_%j.log
# Evidencias do trainer data-parallel MPI (mpi/): teste de invariante (2 ranks x
# 1 rank byte-identicos), refactor puro contra o binario PRE-refactor (arch_d216,
# so' leitura) e o imposto de tau=1. NAO recompila d96/d216 e nao escreve em
# src/build/: o binario MPI vive em mpi/build/ e o scratch em mpi/scratch_20260917/.
#
# REGRA DURA (incidente 2026-09-18 07:18): mpirun SO' dentro do Slurm e NUNCA em
# paralelo com um run longo. O portao abaixo RE-ENFILEIRA enquanto houver outro
# job RUNNING do usuario.
#
# MEMORIA: --mem e' impossivel nesta fermi (slurm.conf reporta RealMemory=1 MB e o
# sbatch recusa qualquer pedido) -> teto por escopo systemd do usuario, via
# mpi/guard.sh, que alem do teto RELATA peak/eventos do cgroup e GRITA em morte
# por memoria. Teto de cada job:
#   mpiddp  6G  (a secao 1 roda o train_run PRE-refactor: RSS fixo ~4,4 GB medido
#               no job 110 -- foi por isso que o job 114 morreu MUDO com 4G)
#   mpithru 6G  / mpiquant 6G  (train_ddp; 4G so' serviria para MPI puro)
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
echo "portao ok: nenhum outro job RUNNING; -c ${SLURM_CPUS_PER_TASK:-?}"

chmod +x mpi/tests_ddp.sh mpi/tests_tax.sh mpi/guard.sh
exec bash mpi/guard.sh 6G \
  nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich \
  -c bash -c 'bash mpi/tests_ddp.sh; bash mpi/tests_tax.sh'
