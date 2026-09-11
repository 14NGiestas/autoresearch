#!/bin/bash
#SBATCH -p compute
#SBATCH -c 10
#SBATCH --mem 8G
#SBATCH -J bpb
#SBATCH -t 00:40:00
#SBATCH -o bpb_%j.log
#
# Held-out-book bpb for several checkpoints, on the quiet box.
# The first entry is a SELF-CHECK: phase-5 step_300 must reproduce the trainer's
# own "val @300 bpb 2.72558" from logs/train_phase5_prose.log. If it does not,
# the aggregation or the row slice is wrong and no other number here is usable.
#
# Usage: sbatch jobs/halfbeast_bpb.sbatch CKPT[:label] [CKPT[:label] ...]
set -u
EV="$HOME/autoresearch-eval"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
cd "$EV/autoresearch" || exit 1

ROWS="$EV/prose_val60.txt"
WIDE="$EV/prose_val300.txt"      # optional wider slice: tighter bpb, ~5x the cost
TB="$EV/tok_tables/token_bytes.txt"
CHAT="$EV/portable/run_chat.sh"

bpb_of() {   # bpb_of <ckpt> <rows-file> <nrows>
    local ck="$1" rows="$2" n="$3"
    [ -x "$EV/portable/bin/eval_bpb" ] || return 1
    # stderr goes to the job log, not /dev/null: a silent failure here cost a
    # whole 30-minute job (missing aggregator, hidden by 2>/dev/null)
    "$EV/portable/run_eval_bpb.sh" --weights "$EV/$ck" --rows "$rows" \
        > "/tmp/bpb_${ck}_$(basename "$rows")" || {
        echo "  (eval_bpb failed for $ck -- see job log)"
        return 1
    }
    python3 scripts/bpb_from_eval.py "/tmp/bpb_${ck}_$(basename "$rows")" "$rows" "$TB" "$n"
}

echo "=== held-out-book bpb | host $(hostname) | job $SLURM_JOB_ID | $(date '+%F %H:%M')"
echo "    rows=$ROWS (60 rows of books never trained)"

# Fail fast: the aggregator lives in scripts/, which is easy to forget when
# syncing jobs/ only -- a run that burns 20 min of eval_bpb and THEN cannot
# aggregate is pure waste (happened in job 6000).
[ -f scripts/bpb_from_eval.py ] || {
    echo "FATAL: scripts/bpb_from_eval.py is not on this box -- rsync scripts/ (not just jobs/) first"
    exit 1
}
for spec in "$@"; do
    ck="${spec%%:*}"
    lbl="${spec##*:}"
    [ "$lbl" = "$ck" ] && lbl="$ck"
    if [ ! -d "$EV/$ck" ]; then
        echo "  $lbl: MISSING $EV/$ck"
        continue
    fi
    # eval_bpb is not part of the portable bundle yet; build-free path: use the
    # same loader via the shipped chat binary is not possible, so fall back to
    # the bundle's eval_bpb if present, else skip with a clear message.
    if [ -x "$EV/portable/bin/eval_bpb" ]; then
        printf '  %-22s %s\n' "$lbl" "$(bpb_of "$ck" "$ROWS" 60)"
    else
        echo "  $lbl: portable eval_bpb not shipped (see bin/sync_eval.sh note)"
    fi
done
if [ -f "$WIDE" ]; then
    echo
    echo "########## wider pass (300 held-out rows, tighter number) ##########"
    for spec in "$@"; do
        ck="${spec%%:*}"; lbl="${spec##*:}"
        [ "$lbl" = "$ck" ] && lbl="$ck"
        [ -d "$EV/$ck" ] || continue
        printf '  %-22s %s\n' "$lbl" "$(bpb_of "$ck" "$WIDE" 300)"
    done
fi
echo "=== done $(date '+%H:%M:%S')"
