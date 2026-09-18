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
# MEMORIA: --mem e' impossivel nesta fermi (RealMemory=1 MB no slurm.conf) ->
# teto via mpi/guard.sh (escopo systemd do usuario), que tambem RELATA peak e
# memory.events do cgroup e GRITA em morte por memoria (o job 114 morreu mudo com
# um teto de 4G abaixo da pegada fixa de ~4,4 GB do trainer). Teto aqui: 6G.
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
echo "portao ok: nenhum outro job RUNNING; -c ${SLURM_CPUS_PER_TASK:-?}"

chmod +x mpi/tests_quant.sh
chmod +x mpi/guard.sh
exec bash mpi/guard.sh 6G \
  nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich \
  -c bash mpi/tests_quant.sh
