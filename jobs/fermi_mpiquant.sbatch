#!/bin/bash
#SBATCH -J mpiquant
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /home/pauli/autoresearch/logs/mpiquant_%j.log
# Desvio do coletivo de 16 bits (--sync-fp16 bf16|fix) contra o run bit-exato:
# desvio numerico dos pesos (max|dw|/|w|, ||dw||/||w||), desvio de qualidade
# (bpb no mesmo holdout de 300 linhas) e o invariante entre ranks com quantizacao.
# Default do app continua --sync-fp16 none (bit-exato); aqui e' que se mede o preco.
#
# MEMORIA: --mem e' impossivel nesta fermi (slurm.conf reporta RealMemory=1 MB e
# o sbatch recusa qualquer pedido) -> teto via escopo systemd do usuario, que faz
# o kernel matar dentro do NOSSO cgroup (CONSTRAINT_MEMCG) em vez de escolher o
# maior processo do no (o global_oom de 2026-09-18 07:18).
# CONCORRENCIA: nunca em paralelo com run longo -- o portao RE-ENFILEIRA.
set -u
cd /home/pauli/autoresearch

OTHERS=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" 2>/dev/null | grep -v "^${SLURM_JOB_ID:-0} " || true)
if [ -n "$OTHERS" ]; then
  echo "portao: outro(s) job(s) RUNNING do usuario -- re-enfileirando:"
  echo "$OTHERS" | sed 's/^/  /'
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
echo "portao ok: nenhum outro job RUNNING; -c ${SLURM_CPUS_PER_TASK:-?}; teto = escopo systemd 4G"

chmod +x mpi/tests_quant.sh
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
exec systemd-run --user --scope --unit "mpiquant-${SLURM_JOB_ID}" \
  -p MemoryMax=4G -p MemorySwapMax=0 \
  nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich \
  -c bash mpi/tests_quant.sh
