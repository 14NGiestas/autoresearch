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

BIN=$(ls -t src/build/live/*/app/repl 2>/dev/null | head -1)
[ -x "$BIN" ] || { echo "no repl binary under src/build/live (build first)" >&2; exit 1; }

CKPT=""
while read -r d; do
    tot=$(ls "$d"/*.npy 2>/dev/null | wc -l)
    nz=$(find "$d" -maxdepth 1 -name '*.npy' -size +0c 2>/dev/null | wc -l)
    if [ "$tot" -gt 0 ] && [ "$nz" -eq "$tot" ]; then
        CKPT="$d"          # keep the newest complete one (sorted)
    fi
done < <(ls -1d "$OUT"/step_* "$OUT"/best 2>/dev/null | sort -t_ -k2 -V)

[ -n "$CKPT" ] || { echo "no complete checkpoint under $OUT" >&2; exit 1; }

export OMP_NUM_THREADS="${REPL_OMP:-4}"
export OPENBLAS_NUM_THREADS="${REPL_OMP:-4}"
echo "repl_now: $CKPT (threads $OMP_NUM_THREADS)" >&2

exec "$BIN" \
    --tables "$HOME/.cache/autoresearch/tok_tables" \
    --weights "$CKPT" \
    --n "${REPL_N:-32}" --temp 0.7 \
    --pres 1.0 --freq 1.0 --rep 1.2 --pwin 64 --plen 0.5 --nblock 3 \
    --stream T --pchunk 64 --stats T "$@"
