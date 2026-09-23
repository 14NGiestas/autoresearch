#!/bin/bash
# mpi/guard.sh -- roda um comando com TETO DE MEMORIA e RELATA o que aconteceu.
#
#   uso: mpi/guard.sh <CAP ex: 6G> <comando...>
#
# Por que existe: o slurm.conf desta fermi reporta RealMemory=1 MB, entao nao ha'
# cgroup de memoria por job (o OOM de 2026-09-18 07:18 foi global_oom). O teto vai
# por `systemd-run --user --scope -p MemoryMax=...` (cgroup v2 do usuario), como no
# scripts/memguard.sh do lab. A DIFERENCA e' a segunda metade: aquele script sai
# mudo quando o kernel mata dentro do cgroup (foi o que aconteceu no job 114: o
# cap de 4G estava abaixo do RSS fixo de 4,4 GB do trainer e a secao 1 terminou
# sem uma linha de explicacao). Aqui o wrapper amostra memory.peak do escopo
# enquanto ele roda, le memory.events (max / oom_kill) e GRITA quando houve morte
# por memoria -- com o teto sugerido.
#
# Numeros medidos: train_run (d216, T=1024) = 4,4 GB de RSS fixo (job 110, plano,
# igual em npy/st); train_ddp -n 2 x 4 threads = ~1,7 GB. Logo: 6G para qualquer
# teste que rode train_run/train_ddp com o nosso modelo; 4G so' para MPI puro.
set -u
CAP=${1:?uso: guard.sh <CAP ex: 6G> <comando...>}; shift
UID_=$(id -u)
SCOPE_ROOT="/sys/fs/cgroup/user.slice/user-${UID_}.slice/user@${UID_}.service"
UNIT="mpiguard-$$-$(date +%s).scope"

systemd-run --user --scope -q --unit "$UNIT" -p MemoryMax="$CAP" -p MemorySwapMax=0 -- "$@" &
pid=$!

# o escopo transitorio vive fora do cgroup do job (app.slice do user manager)
SCOPE=""
for _ in $(seq 1 100); do
  SCOPE=$(find "$SCOPE_ROOT" -maxdepth 3 -type d -name "$UNIT" 2>/dev/null | head -1)
  [ -n "$SCOPE" ] && break
  kill -0 $pid 2>/dev/null || break
  sleep 0.2
done

peak=0; ev_max=0; ev_oom=0
while kill -0 $pid 2>/dev/null; do
  if [ -n "$SCOPE" ] && [ -r "$SCOPE/memory.peak" ]; then
    v=$(cat "$SCOPE/memory.peak" 2>/dev/null || echo 0)
    [ "${v:-0}" -gt "$peak" ] 2>/dev/null && peak=$v
  fi
  if [ -n "$SCOPE" ] && [ -r "$SCOPE/memory.events" ]; then
    v=$(awk '/^max /{print $2}' "$SCOPE/memory.events" 2>/dev/null)
    o=$(awk '/^oom_kill /{print $2}' "$SCOPE/memory.events" 2>/dev/null)
    [ "${v:-0}" -gt "$ev_max" ] 2>/dev/null && ev_max=$v
    [ "${o:-0}" -gt "$ev_oom" ] 2>/dev/null && ev_oom=$o
  fi
  sleep 0.5
done
wait $pid; rc=$?

printf '### guard: cap=%s  peak_observado=%s MB  memory.events(max=%s oom_kill=%s)  rc=%s\n' \
  "$CAP" "$(awk -v p="$peak" 'BEGIN{printf "%.0f", p/1048576}')" "$ev_max" "$ev_oom" "$rc"
if [ "$rc" = 137 ] || [ "$rc" = 9 ] || [ "${ev_oom:-0}" != 0 ] || [ "${ev_max:-0}" != 0 ]; then
  echo "### guard: *** MORTE / ESTOURO POR MEMORIA *** o cgroup passou do teto de $CAP."
  echo "### guard: suba o teto (6G para train_run/train_ddp com o d216; 4G so' para MPI puro)"
  echo "### guard: e confira o log do comando -- as secoes seguintes NAO rodaram."
fi
exit $rc
