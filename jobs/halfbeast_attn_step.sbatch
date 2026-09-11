#!/bin/bash
#SBATCH -p compute
#SBATCH -c 10
#SBATCH --mem 16G
#SBATCH -J attn_step
#SBATCH -t 00:40:00
#SBATCH -o attn_step_%j.log
#
# Does the BLAS attention actually speed up a TRAINING step? Same weights, same
# rows, same flags, only --attn differs; both runs pay identical fixed costs
# (weight load, one val + save at k=nsteps), so the time difference isolates the
# per-step attention work. The final NLL of each run is printed as a sanity
# check: the two paths must train similarly (not bit-identical -- different
# summation order -- but close).
set -u
EV="$HOME/autoresearch-eval"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
cd "$EV/autoresearch" || exit 1

ROWS=/tmp/hb_rows16.txt
head -16 "$EV/prose_val60.txt" > "$ROWS"
CK="$EV/p5_step_300"
NSTEPS="${1:-6}"

echo "=== attention step timing | host $(hostname) | job $SLURM_JOB_ID | $(date '+%H:%M:%S')"
echo "    ckpt $(basename "$CK") | rows 16 | nsteps $NSTEPS | threads $OMP_NUM_THREADS"
for m in naive blas; do
    OUT=/tmp/hb_attn_$m
    rm -rf "$OUT"; mkdir -p "$OUT"
    t0=$(date +%s.%N)
    build/live/*/app/train_run --weights "$CK" --rows "$ROWS" --out "$OUT" \
        --nsteps "$NSTEPS" --lr 1.0e-6 --ntrain 8 --nval 4 --val_every 1000 \
        --save_every 1000 --trn_probe 4 --attn "$m" > "/tmp/hb_${m}.log" 2>&1
    t1=$(date +%s.%N)
    printf '  %-5s total %7.1f s | last nll: %s\n' "$m" "$(echo "$t1 - $t0" | bc)" \
        "$(grep '^step ' "/tmp/hb_${m}.log" | tail -1 | awk '{print $3}')"
    echo "        val @: $(grep -E '^(val|trn) @' "/tmp/hb_${m}.log" | tr '\n' ' ')"
    rm -rf "$OUT"          # do not leave 1.1 GB per mode on a shared box
done
echo "  -> the delta / $NSTEPS is the attention saving per step"
echo "=== done $(date '+%H:%M:%S')"
