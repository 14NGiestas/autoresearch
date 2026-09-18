#!/bin/bash
#SBATCH -J mpithru
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /home/pauli/autoresearch/logs/mpithru_%j.log
# Throughput do train_ddp com --attn blas (default novo), 4 threads/rank,
# lotes disjuntos: -n 1 x -n 2 em tau=1 e tau=100 + estatistica do Allreduce.
#
# MEMORIA (por que NAO ha --mem aqui): nesta fermi o slurm.conf reporta
# RealMemory=1 MB (sinfo: RealMemory=1), entao QUALQUER pedido de memoria e'
# recusado por `sbatch` ("Memory specification can not be satisfied") -- testado
# com --mem=4G e com --mem-per-cpu=512M. O proprio job 110 do supervisor roda com
# ReqTRES=mem=1M. E por isso que o OOM de 2026-09-18 07:18 foi global_oom com
# CONSTRAINT_NONE: nao existe cgroup de memoria para o Slurm cobrar.
# O substituto, que da' a mesma protecao, e' um escopo systemd do usuario:
#   systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 ...
# Com isso o kernel mata DENTRO do nosso cgroup (CONSTRAINT_MEMCG) em vez de
# escolher o maior processo do no (que era o train_run alheio).
#
# CONCORRENCIA: nunca em paralelo com um run longo. O portao abaixo RE-ENFILEIRA
# o job enquanto houver outro job RUNNING do usuario -- o teste MPI espera a
# fermi ficar livre.
set -u
cd /home/pauli/autoresearch

OTHERS=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" 2>/dev/null | grep -v "^${SLURM_JOB_ID:-0} " || true)
if [ -n "$OTHERS" ]; then
  echo "portao: outro(s) job(s) RUNNING do usuario -- re-enfileirando:"
  echo "$OTHERS" | sed 's/^/  /'
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
echo "portao ok: nenhum outro job RUNNING; -c ${SLURM_CPUS_PER_TASK:-?}; teto de memoria = escopo systemd 4G"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
exec systemd-run --user --scope --unit "mpithru-${SLURM_JOB_ID}" \
  -p MemoryMax=4G -p MemorySwapMax=0 \
  nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich \
  -c bash mpi/tests_throughput.sh
