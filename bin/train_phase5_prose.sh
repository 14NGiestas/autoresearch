#!/usr/bin/env bash
# Phase 5 launcher (PROSE-PT): Portuguese books, validation on held-out BOOKS.
#
# Handoff: phase-4 step_1000 (end-of-phase weights AND Adam moments), NOT
# best/ -- best/ is val-selected on the fortran corpus, which says nothing
# about a new distribution, and its moments come from an earlier step
# (Cognivolve: never reset, never rewind the optimizer). Falls back to best/
# or the newest complete step if step_1000 is missing/empty (the phase-3
# disk-full lesson: a directory that exists is not a checkpoint that saved).
#
# Row contract: /tmp/prose/prose_all.txt = 59,385 shuffled train rows (seed
# 20260910) followed by 4,950 val rows from 56 books that are never trained.
#   * --ntrain is the train/val BOUNDARY, not a pool size. The loop walks rows
#     sequentially (r = mod(k-1,ntrain)), so 1000 steps see 1000 DISTINCT rows
#     (2M tokens, 1.7% of the corpus). The instruction phases instead re-read
#     120 rows ~8x, which is what produced verbatim corpus parroting.
#   * --nval 60 + --trn_probe 40: every probe row is a full T=2048 forward, so
#     the default (ntrain-nval = 59,325 rows) would cost ~65 h per validation.
#     The capped probe samples the TAIL of the train pool, i.e. rows this phase
#     does not train on -- a meaningful number instead of memorised ones.
# Logging goes straight to the log file (no tee: the phase-3 tee died with its
# tmux session and froze the log 700 steps early).
set -u

OUT=/tmp/w_fortran                 # phase-4 output (source of the handoff)
POUT=/tmp/w_prose                  # phase-5 output
ROWS=/tmp/prose/prose_all.txt
LOG=/home/pauli/autoresearch/logs/train_phase5_prose.log
TRAIN_BIN=$(ls -t /home/pauli/autoresearch/src/build/live/*/app/train_run 2>/dev/null | head -1)

echo "phase 5 (prose) -> $LOG" >&2

pick_ckpt() {
    local d tot nz
    for d in "$OUT/step_1000" "$OUT/best"; do
        [ -d "$d" ] || continue
        tot=$(ls "$d"/*.npy 2>/dev/null | wc -l)
        nz=$(find "$d" -maxdepth 1 -name '*.npy' -size +0c 2>/dev/null | wc -l)
        if [ "$tot" -eq 90 ] && [ "$nz" -eq 90 ] && [ -f "$d/template.txt" ]; then
            echo "$d"
            return 0
        fi
    done
    for d in $(ls -1d "$OUT"/step_* 2>/dev/null | sort -t_ -k2 -nV -r); do
        tot=$(ls "$d"/*.npy 2>/dev/null | wc -l)
        nz=$(find "$d" -maxdepth 1 -name '*.npy' -size +0c 2>/dev/null | wc -l)
        if [ "$tot" -eq 90 ] && [ "$nz" -eq 90 ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

mkdir -p "$POUT"

{
    echo "=== phase 5 prose — $(date '+%Y-%m-%d %H:%M:%S') ==="
    [ -x "$TRAIN_BIN" ] || { echo "no train_run binary under src/build/live"; exit 1; }
    [ -f "$ROWS" ] || { echo "missing rows file $ROWS"; exit 1; }
    CKPT=$(pick_ckpt) || { echo "no complete checkpoint under $OUT"; exit 1; }
    echo "binary : $TRAIN_BIN"
    echo "weights: $CKPT"
    echo "rows   : $ROWS ($(wc -l < "$ROWS") rows; val starts at row 59385)"
} >>"$LOG" 2>&1

CKPT=$(pick_ckpt) || exit 1

cd /home/pauli/autoresearch/src
exec flock -n "$POUT/train.lock" "$TRAIN_BIN" \
    --weights "$CKPT" \
    --rows "$ROWS" \
    --out "$POUT" \
    --nsteps 1000 --lr 0.00003 \
    --ntrain 59385 --nval 60 --val_every 100 \
    --save_every 100 --keep_last 3 --trn_probe 40 \
    --attn "${PHASE5_ATTN:-blas}" \
    --bytes /home/pauli/.cache/autoresearch/tok_tables/token_bytes.txt \
    >>"$LOG" 2>&1
