#!/bin/bash
#SBATCH -J mpiddp
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /home/pauli/autoresearch/logs/mpiddp_%j.log
# Evidencias do trainer data-parallel MPI (mpi/): teste de invariante (2 ranks x
# 1 rank byte-identicos), refactor puro contra o binario PRE-refactor (arch_d216,
# so leitura) e o imposto de tau=1. NAO recompila d96/d216 e nao escreve em
# src/build/: o binario MPI vive em mpi/build/ e o scratch em mpi/scratch_20260917/.
#
# REGRA DURA (pos-incidente de 2026-09-18 07:18): mpirun/mpiexec NUNCA fora de um
# job do Slurm e NUNCA em paralelo com um run longo. O global_oom daquela manha
# matou o train_run do job 103 (RSS 3.5 GB) quando dois ranks do train_ddp
# (760-845 MB cada) entraram em paralelo numa fermi de 15 GB.
#
# MEMORIA (por que NAO ha --mem aqui): o slurm.conf desta fermi reporta
# RealMemory=1 MB, entao qualquer --mem/--mem-per-cpu e' recusado no sbatch
# ("Memory specification can not be satisfied"). O teto real e' um escopo systemd
# do usuario (MemoryMax=4G), que faz o kernel matar dentro do NOSSO cgroup
# (CONSTRAINT_MEMCG) em vez de escolher o maior processo do no.
set -u
cd /home/pauli/autoresearch

# Portao: se houver QUALQUER outro job RUNNING do mesmo usuario, devolve o job
# para a fila em vez de rodar junto (o portao protege o run alheio; o escopo
# systemd protege contra o OOM global).
OTHERS=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" 2>/dev/null | grep -v "^${SLURM_JOB_ID:-0} " || true)
if [ -n "$OTHERS" ]; then
  echo "portao: outro(s) job(s) RUNNING do usuario -- re-enfileirando:"
  echo "$OTHERS" | sed 's/^/  /'
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
echo "portao ok: nenhum outro job RUNNING; -c ${SLURM_CPUS_PER_TASK:-?}; teto de memoria = escopo systemd 4G"

chmod +x mpi/tests_ddp.sh mpi/tests_tax.sh
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
exec systemd-run --user --scope --unit "mpiddp-${SLURM_JOB_ID}" \
  -p MemoryMax=4G -p MemorySwapMax=0 \
  nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich \
  -c bash -c 'bash mpi/tests_ddp.sh; bash mpi/tests_tax.sh'
