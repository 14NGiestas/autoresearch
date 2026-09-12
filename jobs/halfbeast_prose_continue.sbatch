#!/bin/bash
#SBATCH -p compute
#SBATCH -c 10
#SBATCH --mem 16G
#SBATCH -J prose_cont
#SBATCH -t 06:00:00
#SBATCH -o prose_cont_%j.log
# Continue prose training on ROWS THE MODEL HAS NEVER SEEN.
# The first phase consumed rows 0..999 of the shuffled pool (2M tokens, 1.7%);
# this one starts at row 1000, so more steps mean more data instead of more
# repetitions of the same 1000 rows (which is what the instruction phases did).
# Val stays the same held-out books, so the curve is comparable.
# Usage: sbatch jobs/halfbeast_prose_continue.sbatch <rows> <ntrain> <steps> <out>
set -u
EV="$HOME/autoresearch-eval"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
ROWS="${1:-$EV/prose_v2.txt}"; NTR="${2:-58385}"; NS="${3:-2000}"; OUT="${4:-/tmp/w_prose_v2}"
W="${5:-$EV/p5_blas_best}"   # 5o arg: continuar DE ONDE (senão volta ao pai)
echo "=== prose continue | host $(hostname) | job $SLURM_JOB_ID | $(date '+%F %H:%M')"
echo "    rows=$ROWS ntrain=$NTR steps=$NS out=$OUT from=$W"
rm -rf "$OUT"; mkdir -p "$OUT"
t0=$(date +%s)
"$EV/portable/run_train_run.sh" --weights "$W" --rows "$ROWS" --out "$OUT" \
    --nsteps "$NS" --lr 0.00003 --ntrain "$NTR" --nval 60 --val_every 100 \
    --save_every 100 --keep_last 30 --trn_probe 40 --attn blas \
    --bytes "$EV/tok_tables/token_bytes.txt" > "$EV/prose_cont_run.log" 2>&1
echo "    exit=$? wall=$(( ($(date +%s) - t0) / 60 )) min"
grep -E '^val @' "$EV/prose_cont_run.log" | sed 's/^/      /'
echo "=== done $(date '+%H:%M:%S')"
