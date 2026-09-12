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
# Token budgets: the box is idle, so spend it where length buys signal.
#   core/residue/chat/security/multilingual at N (default 64) -- these answer
#     in a handful of tokens; more is just padding
#   prose at NP (default 256) -- coherence and the novelty metric need real
#     text; at 24 tokens the model emitted 2-5 words and the metric said
#     nothing
#
# Usage: sbatch jobs/halfbeast_gate.sbatch [A] [B] [N] [NP]
set -u
A="${1:-step_1000}"; A="${A%%:*}"      # tolerate the CK:label form
B="${2:-best}";      B="${B%%:*}"
NN="${3:-64}"
NP="${4:-256}"
EV="$HOME/autoresearch-eval"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
cd "$EV/autoresearch" || exit 1

echo "=== prose gate | host $(hostname) | job $SLURM_JOB_ID | $(date '+%F %H:%M')"
echo "    A=$A  B=$B  n=$NN (short batteries)  n=$NP (prose)  threads=$OMP_NUM_THREADS"

run_ck() {
    local ck="$1" tag="$2"
    if [ ! -d "$EV/$ck" ]; then
        echo "MISSING checkpoint $EV/$ck -- skipping $tag"
        return 1
    fi
    echo "--- running $tag ($ck)"
    bin/eval_infer.sh --weights "$EV/$ck" --tables "$EV/tok_tables" \
        --chat "$EV/portable/run_chat.sh" --only core --n "$NN" \
        --label "halfbeast $ck $(date '+%F_%H:%M')" > "$EV/gate_$tag.txt" 2>&1
    for b in residue chat security multilingual; do
        bin/eval_infer.sh --weights "$EV/$ck" --tables "$EV/tok_tables" \
            --chat "$EV/portable/run_chat.sh" --only "$b" --n "$NN" \
            --label "halfbeast $ck $(date '+%F_%H:%M')" >> "$EV/gate_$tag.txt" 2>&1
    done
    bin/eval_infer.sh --weights "$EV/$ck" --tables "$EV/tok_tables" \
        --chat "$EV/portable/run_chat.sh" --only prose --n "$NP" \
        --label "halfbeast $ck $(date '+%F_%H:%M')" > "$EV/gate_${tag}_prose.txt" 2>&1
    cat "$EV/gate_${tag}_prose.txt" >> "$EV/gate_$tag.txt"
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
