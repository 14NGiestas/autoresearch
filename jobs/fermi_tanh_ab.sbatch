#!/bin/bash
#SBATCH -J tanhab
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /home/pauli/autoresearch/logs/tanhab_%j.log
# Does the fast tanh move the speed estimate? A/B, same model, same cap.
#
# The soft cap evaluates one tanh per score pair, which is T^2/2 per (batch,
# head, row). At T=1024 with 6 heads and 12 layers that is about 38 million calls
# per step, so the cost of the tanh is not a detail. The polynomial has 8 terms
# and no branch; the intrinsic tanh has branches and a libm call.
#
# Two binaries, one switch: -DLOGIT_CAP_EXACT_TANH swaps the polynomial for the
# intrinsic. Quality is already measured (2e-5 to 2.2e-4 nll, so the estimate of
# bpb does not move). This job measures the time.
#
# The idle guard is not optional: a step measurement next to another job is a
# corrupted measurement. Three samples of load below 1.5 are required first.
set -u
cd /home/pauli/autoresearch
export OMP_NUM_THREADS=8
PY=.venv-numpy/bin/python3
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
CK=/tmp/dl/122/round16_common
STEPS=20
echo "tanhab job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"

# idle guard: wait for a quiet node before any timed section
for t in $(seq 1 120); do
  L=$(cut -d' ' -f1 /proc/loadavg)
  if [ "$(echo "$L < 1.5" | bc -l)" = "1" ]; then
    N=$(( ${N:-0} + 1 ))
    [ "$N" -ge 3 ] && { echo "  node ocioso (load=$L, 3 amostras)"; break; }
  else
    N=0
  fi
  sleep 20
done
[ "${N:-0}" -lt 3 ] && { echo "  DESISTI: o node nunca ficou ocioso (load=$L)"; exit 3; }

printf "%-8s %-8s %-10s %-10s %-12s\n" "cap" "tanh" "wall_s" "s/step" "nll"
run () {  # cap mode
  local CAP=$1 MODE=$2
  local BIN=$(ls /tmp/sscap/ab_$MODE/*/app/train_run | tail -1)
  local T0=$(date +%s.%N)
  local OUT=$($BIN --weights $CK --rows /tmp/mix/rows_f0.npy --out /tmp/sscap/tb_${MODE}_$CAP \
      --nsteps $STEPS --lr 0 --ntrain 11797 --start_row 0 --nval 1 --val_every 9999999 \
      --trn_probe 1 --save_every 99 --attn blas --batch 1 --bytes "$BY" --logit-cap $CAP 2>&1)
  local T1=$(date +%s.%N)
  local W=$(echo "$T1-$T0" | bc -l)
  local SP=$(echo "$W/$STEPS" | bc -l | cut -c1-5)
  local L=$(echo "$OUT" | grep '^step' | tail -1 | awk '{print $4}')
  printf "%-8s %-8s %-10.2f %-10s %-12s\n" "$CAP" "$MODE" "$W" "$SP" "${L:-?}"
}
for CAP in 0 2 30; do
  run $CAP fast
  run $CAP exact
done
echo "fim tanhab $(date '+%F %H:%M:%S')"
