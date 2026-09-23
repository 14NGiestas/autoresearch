#!/usr/bin/env bash
# bin/repl_now.sh — interactive repl on the newest COMPLETE checkpoint.
#
# Encodes two lessons that each cost real debugging time:
#
#  1. train_run pre-creates every step_N directory at launch (all of them,
#     empty). So "newest checkpoint dir" is usually an empty placeholder:
#     phase-3 step_300/step_400 looked saved and were 0-byte. Pick the newest
#     directory whose .npy files are ALL present and non-empty.
#
#  2. Interactive inference must NOT inherit the trainer's thread counts.
#     With no OMP_NUM_THREADS the repl opens 16 OpenMP + 16 OpenBLAS threads
#     against a 16-thread training run: 32 threads on 16 cores, and a T=1
#     GEMV per token is pure thread-launch overhead. Measured on the same
#     prompt/weights: 0.87 tok/s unset vs 36.9 tok/s at 4/4 (42x).
#
# Usage: bin/repl_now.sh [OUT_DIR] [extra repl flags...]
#   OUT_DIR default /tmp/w_fortran; REPL_OMP / REPL_N override threads / tokens.
set -u
OUT="${1:-/tmp/w_fortran}"
shift || true

CKPT=""
while read -r d; do
    tot=$(ls "$d"/*.npy 2>/dev/null | wc -l)
    nz=$(find "$d" -maxdepth 1 -name '*.npy' -size +0c 2>/dev/null | wc -l)
    if [ "$tot" -gt 0 ] && [ "$nz" -eq "$tot" ]; then
        CKPT="$d"          # keep the newest complete one (sorted)
    fi
done < <(ls -1d "$OUT"/step_* "$OUT"/best 2>/dev/null | sort -t_ -k2 -V)

# accept a bare checkpoint dir too ($OUT itself holds complete .npy files)
if [ -z "$CKPT" ]; then
    tot=$(ls "$OUT"/*.npy 2>/dev/null | wc -l)
    nz=$(find "$OUT" -maxdepth 1 -name '*.npy' -size +0c 2>/dev/null | wc -l)
    if [ "$tot" -gt 0 ] && [ "$nz" -eq "$tot" ]; then CKPT="$OUT"; fi
fi

[ -n "$CKPT" ] || { echo "no complete checkpoint under $OUT" >&2; exit 1; }

BIN=""
# 1. Tenta descobrir a arquitetura pelo arch.txt do checkpoint
if [ -f "$CKPT/arch.txt" ]; then
    # Formato esperado: D_MODEL=216 N_HEAD=6 N_KV=2 N_LAYER=12 VV=8192 TT=1024
    # Queremos transformar em: arch_d216_h6_kv2_l12_v8192_c1024
    # arch.txt real: "d_model = 216" (minusculo, com espacos). Normaliza tudo.
    eval "$(sed -E 's/ *//g' "$CKPT/arch.txt" | grep -E '^[a-z_]+=[0-9]+' | sed 's/d_model/d/; s/n_head/h/; s/n_kv/kv/; s/n_layer/l/; s/vocab/v/; s/ctx/c/')"
    ARCH_DIR="arch_d${d}_h${h}_kv${kv}_l${l}_v${v}_c${c}"
    BIN=$(ls -t "src/build/$ARCH_DIR"/*/app/repl 2>/dev/null | head -1)
    
    # Heurística para o --space: se V=8192, é o nosso BPE v1.
    [ "${v:-0}" -eq 8192 ] && SPACE_FLAG="--space bpe" || SPACE_FLAG="--space byte"
fi

# 2. Fallback para build/live se não achou específico ou não tem arch.txt
if [ -z "$BIN" ]; then
    BIN=$(ls -t src/build/live/*/app/repl 2>/dev/null | head -1)
    SPACE_FLAG="--space bpe" # default seguro
fi

[ -x "$BIN" ] || { echo "no repl binary found for $CKPT (checked specific arch and live)" >&2; exit 1; }

export OMP_NUM_THREADS="${REPL_OMP:-4}"
export OPENBLAS_NUM_THREADS="${REPL_OMP:-4}"
export OMP_DYNAMIC=FALSE
echo "repl_now: $CKPT ($SPACE_FLAG, threads $OMP_NUM_THREADS)" >&2

exec "$BIN" \
    --tables "$HOME/.cache/autoresearch/tok_tables" \
    --weights "$CKPT" \
    $SPACE_FLAG \
    --n "${REPL_N:-128}" --temp 0.7 \
    --pres 1.1 --freq 1.1 --rep 1.2 --pwin 64 --plen 0.5 --nblock 3 \
    --stream T --pchunk 64 --stats T "$@"
