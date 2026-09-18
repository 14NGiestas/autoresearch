#!/bin/bash
#SBATCH -J mpithru
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /home/pauli/autoresearch/logs/mpithru_%j.log
# Throughput do train_ddp com --attn blas (default novo), 4 threads/rank,
# lotes disjuntos: -n 1 x -n 2 em tau=1 e tau=100 + estatistica do Allreduce.
#
# MEMORIA: nesta fermi o slurm.conf reporta RealMemory=1 MB, entao QUALQUER
# pedido de --mem/--mem-per-cpu e' recusado pelo sbatch (o proprio job 110 do
# supervisor roda com ReqTRES=mem=1M) e o OOM de 07:18 foi global_oom sem cgroup.
# O substituto e' mpi/guard.sh (escopo systemd do usuario) que, alem do teto,
# RELATA peak/eventos do cgroup e GRITA em morte por memoria. Teto 6G: o job 114
# morreu MUDO com 4G porque a pegada fixa do trainer e' ~4,4 GB.
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
echo "portao ok: nenhum outro job RUNNING; -c ${SLURM_CPUS_PER_TASK:-?}"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
chmod +x mpi/guard.sh
exec bash mpi/guard.sh 6G \
  nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich \
  -c bash mpi/tests_throughput.sh
