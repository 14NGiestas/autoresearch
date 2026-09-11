#!/bin/bash
#SBATCH -p compute
#SBATCH -c 10
#SBATCH --mem 8G
#SBATCH -J batteries
#SBATCH -t 00:40:00
#SBATCH -o batteries_%j.log
#
# Run the full phased battery suite against one checkpoint, on the quiet box.
# Usage: sbatch jobs/halfbeast_batteries.sbatch <ckpt-name> [n-tokens]
#   e.g. sbatch jobs/halfbeast_batteries.sbatch step_1000 24
#
# Threads: 10 = physical cores (measured optimum; hyperthreading costs ~10% on
# decode). Portable bundle = fermi-built binary + its .so closure, md5-verified
# identical to fermi's outputs.
set -u
CK="${1:?usage: sbatch halfbeast_batteries.sbatch CKPT [n]}"
NN="${2:-24}"
EV="$HOME/autoresearch-eval"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
cd "$EV/autoresearch" || exit 1

[ -d "$EV/$CK" ] || { echo "no checkpoint $EV/$CK"; exit 1; }
nz=$(find "$EV/$CK" -maxdepth 1 -name '*.npy' -size +0c | wc -l)

echo "=== host $(hostname) | job $SLURM_JOB_ID | ckpt $CK ($nz/90 files) | n=$NN | $(date '+%F %H:%M')"
bin/eval_infer.sh --weights "$EV/$CK" --tables "$EV/tok_tables" \
    --chat "$EV/portable/run_chat.sh" --only all --n "$NN" \
    --label "halfbeast $CK $(date '+%F_%H:%M')" > "$EV/batteries_$CK.txt" 2>&1

echo "cases: $(grep -c '^=== ' "$EV/batteries_$CK.txt")  lines: $(wc -l < "$EV/batteries_$CK.txt")"
echo
echo "--- in-dist / chat / prose highlights ---"
grep -A1 -E '^=== (in-dist|template|ood/greeting)' "$EV/batteries_$CK.txt" | head -22
echo
echo "--- prose battery metrics ---"
grep -E '^=== prose/|\[metric\]' "$EV/batteries_$CK.txt" | paste - - | head -8
echo "=== done $(date '+%H:%M:%S')"
