#!/usr/bin/env bash
# train_chunked.sh — run longo em SEGMENTOS encadeados, com teto de memoria.
#
# Por que existe: o job 103 (63.889 passos, 7h40) morreu em global_oom com
# train_run em 3,5 GB de RSS — 13x o tamanho do arquivo de rows e ~50x o que um
# d96 deveria ocupar. Run curto nunca morreu; so o longo. Ate o vazamento ser
# localizado (job 110), a defesa e nao deixar o processo viver o suficiente para
# estourar: cada segmento re-executa o binario, entao o RSS volta ao piso.
#
# A segmentacao e EXATA, nao aproximada: com LR constante, continuar de um
# checkpoint com --t0 <passo+1> --start_row <passo mod ntrain> --ntrain <pool>
# reproduz o mesmo stream de dados e a mesma numeracao de diretorios que o run
# unico teria (verificado no job 109: os step_63488/63889 batem com o previsto).
#
# Uso:
#   scripts/train_chunked.sh --seg 20000 --vlimit-gb 6 -- <args do train_run>
# (os args precisam conter --out, --nsteps, --start_row, --ntrain, --save_every)
set -u
SEG=20000; VLIM=6; args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --seg) SEG=$2; shift 2;;
    --vlimit-gb) VLIM=$2; shift 2;;
    --) shift; args=("$@"); break;;
    *) echo "uso: $0 [--seg N] [--vlimit-gb G] -- <args>"; exit 2;;
  esac
done
[ ${#args[@]} -gt 0 ] || { echo "faltam os args do train_run"; exit 2; }
BIN=$(ls -t "$(dirname "$0")/../src/build"/arch_*/gfortran_*/app/train_run 2>/dev/null | head -1)
[ -x "$BIN" ] || { echo "train_run nao encontrado"; exit 2; }
get() { for i in "${!args[@]}"; do [ "${args[$i]}" = "$1" ] && { echo "${args[$((i+1))]}"; return; }; done; }
OUT=$(get --out); N=$(get --nsteps); SROW=$(get --start_row); NTRAIN=$(get --ntrain); T0=$(get --t0)
T0=${T0:-1}
[ -n "$OUT" ] && [ -n "$N" ] || { echo "--out e --nsteps sao obrigatorios"; exit 2; }
done_=0
while [ $done_ -lt $N ]; do
  rem=$((N - done_)); take=$((rem < SEG ? rem : SEG))
  sr=$(( (SROW + done_) % NTRAIN ))
  # args sem --nsteps/--start_row/--t0/--out (reconstruidos por segmento)
  seg=()
  skip=0
  for i in "${!args[@]}"; do
    if [ $skip -eq 1 ]; then skip=0; continue; fi
    case "${args[$i]}" in
      --nsteps|--start_row|--t0|--out) skip=1;;
      *) seg+=("${args[$i]}");;
    esac
  done
  echo "[chunk] passo $((T0+done_))..$((T0+done_+take-1)) start_row=$sr"
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OMP_DYNAMIC=FALSE \
    "$(dirname "$0")/memguard.sh" ${VLIM}G "$BIN" \
      "${seg[@]}" --out "$OUT" --nsteps "$take" --start_row "$sr" \
      --ntrain "$NTRAIN" --t0 $((T0+done_)) || {
        echo "[chunk] segmento falhou (vlimit ${VLIM}G) em $((T0+done_)) — abortando"; exit 1; }
  done_=$((done_+take))
done
echo "[chunk] ok: $N passos em segmentos de $SEG (vlimit ${VLIM}G)"
