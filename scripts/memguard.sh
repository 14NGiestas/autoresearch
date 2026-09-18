#!/usr/bin/env bash
# memguard.sh — roda um comando com teto de memoria REAL, driblando o Slurm da fermi.
#
# O slurm.conf desta maquina reporta RealMemory=1 MB, entao --mem e --mem-per-cpu
# sao RECUSADOS e nenhum job tem cgroup de memoria: quando a maquina estoura, o
# kernel escolhe a vitima por tamanho (global_oom), e foi assim que perdemos o
# job 103 (train_run de 3.5 GB morto para salvar o smoke test MPI alheio).
#
# Aqui o teto vai por systemd-run --user --scope -p MemoryMax=... (o kernel passa a
# matar dentro do NOSSO cgroup) e, se nao houver user manager, cai para ulimit -v.
# Uso: memguard.sh 6G <comando...>
set -u
LIM=${1:?uso: memguard.sh <ex: 6G> <comando...>}; shift
if systemd-run --user --scope -q -p MemoryMax=$LIM -p MemorySwapMax=0 -- "$@" 2>/dev/null; then
  exit 0
fi
case "$LIM" in
  *G|*g) KB=$(( ${LIM%[Gg]} * 1024 * 1024 ));;
  *M|*m) KB=$(( ${LIM%[Mm]} * 1024 ));;
  *)     KB=$LIM;;
esac
( ulimit -v "$KB" 2>/dev/null || true; "$@" )
