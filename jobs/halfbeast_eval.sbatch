#!/bin/bash
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem 8G
#SBATCH -J eval_portable
#SBATCH -t 00:40:00
#SBATCH -o eval_portable_%j.log
#
# Clean-box re-measurement on halfbeast (i9-7900X, 10c/20t, idle).
#
# Why: every number measured on fermi today was taken next to a 16-thread
# trainer -- identical configs varied by +-20%, and the spec/prefill ratios
# were honest but not quotable. This job re-runs them on an idle machine.
#
# How the binary gets here without sudo or a system change on a shared box:
# the portable bundle = fermi-built chat_text + the exact .so files it needs
# (same OpenBLAS, same glibc) + a loader wrapper. Same arithmetic as fermi,
# only the CPU differs, and OpenBLAS dispatches kernels per CPU at runtime.
# The md5 checks below assert that claim instead of trusting it.
set -u
EV="$HOME/autoresearch-eval"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
CHAT="$EV/portable/run_chat.sh"
W="$EV/step_300"
T="$EV/tok_tables"
cd "$EV/autoresearch" || exit 1

echo "=== host $(hostname) | job $SLURM_JOB_ID | $(date '+%F %H:%M')"
echo "=== cpus: ${SLURM_CPUS_ON_NODE:-?} | threads: OMP=$OMP_NUM_THREADS"

python3 - <<'PY'
s = "Find the GCD of 54 and 24. Determine whether 97 is prime. What is the LCM of 12 and 18? "
open('/tmp/hb_long.txt', 'w').write(s * 40)                                  # ~880 tokens
open('/tmp/hb_rep.txt', 'w').write("Find the GCD of 54 and 24. Result: 6. " * 20)
PY

echo
echo "########## 1. all six batteries (portable binary) ##########"
bin/eval_infer.sh --weights "$W" --tables "$T" --chat "$CHAT" --only all --n 24 \
    --label "halfbeast $(hostname -s) $(date '+%F_%H:%M')" | tee "$EV/batteries.txt" | tail -5
echo "batteries.txt: $(wc -l < "$EV/batteries.txt") lines"

echo
echo "########## 2. prefill sweep (--pchunk), 880-token prompt, n=4 ##########"
for pc in 1 64 256; do
    printf '%s' "$(cat /tmp/hb_long.txt)" | "$CHAT" --tables "$T" --weights "$W" \
        --n 4 --temp 0.0 --template raw --stream F --stats T --pchunk "$pc" \
        > "/tmp/hb_pc_$pc.out" 2> "/tmp/hb_pc_$pc.err"
    printf '  pchunk=%-4s %s  md5=%s\n' "$pc" \
        "$(grep -o 'prefill_ms=[ ]*[0-9.]*' "/tmp/hb_pc_$pc.err" | tr -d ' ')" \
        "$(md5sum < "/tmp/hb_pc_$pc.out" | cut -c1-12)"
done
echo "  (equivalence: the three md5 must be identical)"

echo
echo "########## 3. spec-decode sweep (--spec), repetitive prompt, n=48 ##########"
for sp in 0 8; do
    printf '%s' "$(cat /tmp/hb_rep.txt)" | "$CHAT" --tables "$T" --weights "$W" \
        --n 48 --temp 0.0 --template raw --stream F --stats T --spec "$sp" \
        > "/tmp/hb_sp_$sp.out" 2> "/tmp/hb_sp_$sp.err"
    printf '  spec=%-2s %s | %s | md5=%s\n' "$sp" \
        "$(grep -o 'decode_ms=[ ]*[0-9.]*' "/tmp/hb_sp_$sp.err" | tr -d ' ')" \
        "$(grep -o 'spec:.*' "/tmp/hb_sp_$sp.err" | head -1)" \
        "$(md5sum < "/tmp/hb_sp_$sp.out" | cut -c1-12)"
done
echo "  (equivalence: both md5 must be identical)"

echo
echo "########## done $(date '+%H:%M:%S') ##########"
