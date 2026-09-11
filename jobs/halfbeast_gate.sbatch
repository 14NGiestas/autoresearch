#!/bin/bash
#SBATCH -p compute
#SBATCH -c 10
#SBATCH --mem 12G
#SBATCH -J prose_gate
#SBATCH -t 00:40:00
#SBATCH -o prose_gate_%j.log
#
# The phase-5 gate. Compares two checkpoints on the SAME battery suite:
#   A = phase-4 step_1000  (the baseline: pure fortran corpus, core battery
#       already degraded -- in-dist gcd empty, prime a row of dots)
#   B = phase-5 best       (pure Portuguese prose, val bpb 3.07 -> 2.73)
# so the comparison is like-for-like: same binary (portable bundle), same
# thread count (10 = physical cores), same prompts/seeds.
#
# n=64, not 24: at 24 tokens the mid-training model emits 2-5 words and the
# prose novelty metric is meaningless.
#
# Falsifiable claims (hyp_1fbfd5 / hyp_5ca396):
#   * prose + multilingual improve over A        -> PROSE-PT gains
#   * core does NOT fall below A                 -> no extra forgetting
#   * if core is still degraded, the repair is a MIXED phase, not the order
#
# Usage: sbatch jobs/halfbeast_gate.sbatch [A] [B] [n]
set -u
A="${1:-step_1000}"
B="${2:-best}"
NN="${3:-64}"
EV="$HOME/autoresearch-eval"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
cd "$EV/autoresearch" || exit 1

echo "=== prose gate | host $(hostname) | job $SLURM_JOB_ID | $(date '+%F %H:%M')"
echo "    A=$A  B=$B  n=$NN  threads=$OMP_NUM_THREADS"

run_ck() {
    local ck="$1" tag="$2"
    if [ ! -d "$EV/$ck" ]; then
        echo "MISSING checkpoint $EV/$ck -- skipping $tag"
        return 1
    fi
    echo "--- running $tag ($ck)"
    bin/eval_infer.sh --weights "$EV/$ck" --tables "$EV/tok_tables" \
        --chat "$EV/portable/run_chat.sh" --only all --n "$NN" \
        --label "halfbeast $ck $(date '+%F_%H:%M')" > "$EV/gate_$tag.txt" 2>&1
    echo "    $(grep -c '^=== ' "$EV/gate_$tag.txt") cases -> $EV/gate_$tag.txt"
}

run_ck "$A" phase4
run_ck "$B" phase5

echo
echo "########## prose battery: side by side (metric lines) ##########"
for tag in phase4 phase5; do
    [ -f "$EV/gate_$tag.txt" ] || continue
    echo "--- $tag"
    grep -E '^=== prose/|\[metric\]' "$EV/gate_$tag.txt" | paste - - | sed 's/^/    /'
done

echo
echo "########## core battery: first output line per case ##########"
for tag in phase4 phase5; do
    [ -f "$EV/gate_$tag.txt" ] || continue
    echo "--- $tag"
    awk '/^=== (in-dist|paraphrase|magnitude)/{lbl=$0; getline; printf "    %-42s %s\n", lbl, (length($0)?substr($0,1,40):"<vazio>")}' \
        "$EV/gate_$tag.txt"
done

echo
echo "########## multilingual (the hypothesis' sharpest claim) ##########"
for tag in phase4 phase5; do
    [ -f "$EV/gate_$tag.txt" ] || continue
    echo "--- $tag"
    awk '/^=== multi\//{lbl=$0; getline; printf "    %-30s %s\n", lbl, (length($0)?substr($0,1,44):"<vazio>")}' \
        "$EV/gate_$tag.txt"
done
echo
echo "=== note: held-out-book bpb for both checkpoints is separate -- ship a"
echo "    small val slice and run eval_bpb, or read the trainer's val @N lines."
echo "=== done $(date '+%H:%M:%S')"
