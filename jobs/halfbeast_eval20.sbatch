#!/bin/bash
#SBATCH -p compute
#SBATCH -c 20
#SBATCH --mem 12G
#SBATCH -J eval_t20
#SBATCH -t 00:20:00
#SBATCH -o eval_t20_%j.log
#
# Thread-scaling measurement (the question fermis's noisy box could never
# answer): does hyperthreading help or hurt?
#
# Two regimes that should behave differently, and it is worth knowing which
# one we are in:
#   * chunked prefill  = GEMM over TB tokens -> compute bound -> more threads
#     should keep helping (up to the 10 physical cores, then HT may add a bit)
#   * single-token decode = GEMV per token -> memory-bandwidth bound -> more
#     threads usually buy nothing, and HT can hurt (two threads per core
#     fighting for the same caches and load ports)
# Node: i9-7900X, 10 cores / 20 threads, CPUTot=20.
set -u
EV="$HOME/autoresearch-eval"
CHAT="$EV/portable/run_chat.sh"
W="$EV/step_300"
T="$EV/tok_tables"
export OMP_DYNAMIC=FALSE

echo "=== host $(hostname) | job $SLURM_JOB_ID | cpus ${SLURM_CPUS_ON_NODE:-?} | $(date '+%H:%M:%S')"
python3 - <<'PY'
s = "Find the GCD of 54 and 24. Determine whether 97 is prime. What is the LCM of 12 and 18? "
open('/tmp/hb_long.txt', 'w').write(s * 40)
open('/tmp/hb_rep.txt', 'w').write("Find the GCD of 54 and 24. Result: 6. " * 20)
PY

echo
echo "########## chunked prefill (--pchunk 64, 880-token prompt, n=4) ##########"
for TH in 2 4 8 10 16 20; do
    export OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH
    printf '%s' "$(cat /tmp/hb_long.txt)" | "$CHAT" --tables "$T" --weights "$W" \
        --n 4 --temp 0.0 --template raw --stream F --stats T --pchunk 64 \
        > /tmp/t20_pre_$TH.out 2> /tmp/t20_pre_$TH.err
    printf '  threads=%-3s %-22s md5=%s\n' "$TH" \
        "$(grep -o 'prefill_ms=[ ]*[0-9.]*' /tmp/t20_pre_$TH.err | tr -d ' ')" \
        "$(md5sum < /tmp/t20_pre_$TH.out | cut -c1-12)"
done

echo
echo "########## single-token decode (--spec 0, n=48, repetitive prompt) ##########"
for TH in 2 4 8 10 16 20; do
    export OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH
    printf '%s' "$(cat /tmp/hb_rep.txt)" | "$CHAT" --tables "$T" --weights "$W" \
        --n 48 --temp 0.0 --template raw --stream F --stats T --spec 0 \
        > /tmp/t20_dec_$TH.out 2> /tmp/t20_dec_$TH.err
    printf '  threads=%-3s %-21s md5=%s\n' "$TH" \
        "$(grep -o 'decode_ms=[ ]*[0-9.]*' /tmp/t20_dec_$TH.err | tr -d ' ')" \
        "$(md5sum < /tmp/t20_dec_$TH.out | cut -c1-12)"
done

echo
echo "########## speculative decode (--spec 8, same prompt) ##########"
for TH in 4 10 20; do
    export OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH
    printf '%s' "$(cat /tmp/hb_rep.txt)" | "$CHAT" --tables "$T" --weights "$W" \
        --n 48 --temp 0.0 --template raw --stream F --stats T --spec 8 \
        > /tmp/t20_sp_$TH.out 2> /tmp/t20_sp_$TH.err
    printf '  threads=%-3s %-21s %-46s md5=%s\n' "$TH" \
        "$(grep -o 'decode_ms=[ ]*[0-9.]*' /tmp/t20_sp_$TH.err | tr -d ' ')" \
        "$(grep -o 'spec:.*' /tmp/t20_sp_$TH.err | head -1)" \
        "$(md5sum < /tmp/t20_sp_$TH.out | cut -c1-12)"
done
echo "=== done $(date '+%H:%M:%S')"
