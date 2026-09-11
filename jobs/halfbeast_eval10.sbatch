#!/bin/bash
#SBATCH -p compute
#SBATCH -c 10
#SBATCH --mem 8G
#SBATCH -J eval_t10
#SBATCH -t 00:20:00
#SBATCH -o eval_t10_%j.log
#
# 10-thread re-run of the sweeps (halfbeast has 10 physical cores; the box was
# idle, load 0.04, so half of it is polite and useful: chunked prefill is a
# GEMM and scales with cores, while single-token decode is memory-bandwidth
# bound and should NOT scale -- this job shows both).
#
# It also prints a determinism fingerprint: the same prompts/seeds as run on
# fermi, md5'd. Equal md5 across two different CPUs (Zen 4 vs Skylake-X, with
# OpenBLAS dispatching different GEMM kernels) is the cross-machine
# correctness claim; any difference is ISA/BLAS, not noise.
set -u
EV="$HOME/autoresearch-eval"
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
CHAT="$EV/portable/run_chat.sh"
W="$EV/step_300"
T="$EV/tok_tables"

echo "=== host $(hostname) | job $SLURM_JOB_ID | cpus ${SLURM_CPUS_ON_NODE:-?} | OMP=$OMP_NUM_THREADS"
python3 - <<'PY'
s = "Find the GCD of 54 and 24. Determine whether 97 is prime. What is the LCM of 12 and 18? "
open('/tmp/hb_long.txt', 'w').write(s * 40)
open('/tmp/hb_rep.txt', 'w').write("Find the GCD of 54 and 24. Result: 6. " * 20)
PY

echo
echo "########## prefill sweep @10 threads ##########"
for pc in 1 64 256; do
    printf '%s' "$(cat /tmp/hb_long.txt)" | "$CHAT" --tables "$T" --weights "$W" \
        --n 4 --temp 0.0 --template raw --stream F --stats T --pchunk "$pc" \
        > "/tmp/hb10_pc_$pc.out" 2> "/tmp/hb10_pc_$pc.err"
    printf '  pchunk=%-4s %s  md5=%s\n' "$pc" \
        "$(grep -o 'prefill_ms=[ ]*[0-9.]*' "/tmp/hb10_pc_$pc.err" | tr -d ' ')" \
        "$(md5sum < "/tmp/hb10_pc_$pc.out" | cut -c1-12)"
done

echo
echo "########## spec sweep @10 threads ##########"
for sp in 0 8; do
    printf '%s' "$(cat /tmp/hb_rep.txt)" | "$CHAT" --tables "$T" --weights "$W" \
        --n 48 --temp 0.0 --template raw --stream F --stats T --spec "$sp" \
        > "/tmp/hb10_sp_$sp.out" 2> "/tmp/hb10_sp_$sp.err"
    printf '  spec=%-2s %s | %s | md5=%s\n' "$sp" \
        "$(grep -o 'decode_ms=[ ]*[0-9.]*' "/tmp/hb10_sp_$sp.err" | tr -d ' ')" \
        "$(grep -o 'spec:.*' "/tmp/hb10_sp_$sp.err" | head -1)" \
        "$(md5sum < "/tmp/hb10_sp_$sp.out" | cut -c1-12)"
done

echo
echo "########## determinism fingerprint (compare with fermi) ##########"
fp() { printf '%s' "$1" | "$CHAT" --tables "$T" --weights "$W" --n "$2" --temp 0.0 \
        --seed "$3" --template raw --stream F 2>/dev/null | md5sum | cut -c1-12; }
echo "  fp_in_dist_gcd    $(fp 'Find the GCD of 54 and 24.' 16 7)"
echo "  fp_magnitude      $(fp 'Find the GCD of 123456 and 789012.' 16 7)"
echo "  fp_prose_machado  $(fp 'O assunto deste poema é rigorosamente histórico. Em 1659, era prelado administrador do Rio de Janeiro' 24 7)"
echo "  fp_en_control     $(fp 'It is a truth universally acknowledged, that a single man in possession of a good fortune' 24 7)"
echo "=== done $(date '+%H:%M:%S')"
