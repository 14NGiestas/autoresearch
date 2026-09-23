#!/bin/bash
#SBATCH -J archqual
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o /home/pauli/autoresearch/logs/archqual_%j.log
# Quality sweep for head_dim. The step got 1.8x to 2.4x faster, but that test
# had three confounds: kv changed with the head count, one seed, and 1M tokens.
# This job fixes kv at 1 (MQA) and changes only head_dim.
#
# Arms: (6,2,16) is our model today. (6,1,16) separates MQA from GQA at one
# head_dim. (3,1,32) and (2,1,48) then move head_dim alone. Seed 2 repeats the
# two decisive arms, so the noise is measured and not assumed.
#
# The FLOPs stay the same in every arm. The KV bytes per token change, because
# the KV width follows head_dim. Report that number with the result.
set -u
cd /home/pauli/autoresearch
export OMP_NUM_THREADS=8
PY=.venv-numpy/bin/python3
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
ROWS=/tmp/mix/rows_f0.npy
HOLD=/tmp/mix/rows_holdout.npy
STEPS=2929
echo "archqual job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"
printf "%-16s %-9s %-11s %-9s %-11s %-8s\n" "arm" "head_dim" "kv_B/token" "s/step" "bpb" "texture"

run_arm () {   # heads kv seed
  local H=$1 KV=$2 SEED=$3
  local TAG="h${H}_kv${KV}_s${SEED}"
  local OUT=/tmp/aq/$TAG
  local BD=/tmp/aq/build_h${H}_kv${KV}
  local HD=$((96/H))
  local KVB=$((12 * 2 * KV * HD * 4))
  rm -rf "$OUT"
  $PY scripts/make_init.py --out "$OUT" --d 96 --heads "$H" --kv "$KV" --layers 12 \
      --vocab 8192 --ctx 1024 --seed "$SEED" >/dev/null || { echo "  $TAG: init falhou"; return; }
  if [ ! -x "$(ls $BD/*/app/train_run 2>/dev/null | tail -1 || true)" ]; then
    bin/build_arch 96 "$H" "$KV" 12 8192 1024 "$BD" > "$BD.log" 2>&1 || { echo "  $TAG: build falhou"; return; }
  fi
  bin/job_env --build-dir "$BD" "--rows:src/app/train_run.f90" >/dev/null || exit 2
  . "$BD/job_env.sh"
  local BIN="$TRAIN_RUN"
  local EVB="$EVAL_BPB"
  local RPL=$(ls $BD/*/app/repl | tail -1)
  local T0=$(date +%s.%N)
  $BIN --weights "$OUT" --rows $ROWS --out "$OUT/run" --nsteps $STEPS --lr 6e-4 \
      --ntrain 11797 --start_row 0 --nval 1 --val_every 9999999 --trn_probe 1 \
      --save_every $STEPS --attn blas --batch 1 --bytes "$BY" > "$OUT/train.log" 2>&1
  local T1=$(date +%s.%N)
  local SP=$(echo "($T1-$T0)/$STEPS" | bc -l | cut -c1-5)
  local CK=$(ls -d "$OUT"/run/step_* 2>/dev/null | tail -1)
  local B=$($EVB --weights "$CK" --rows $HOLD --batch 16 --attn blas 2>/dev/null | grep -oE "[0-9]\.[0-9]{4,}" | tail -1)
  # A texture reading on the model's own output. Cheap, and it ties this sweep to
  # the texture work.
  local TX="n/d"
  if [ -n "$RPL" ]; then
    printf 'O que e a computacao?\n' | timeout 300 "$RPL" --tables ~/.cache/autoresearch/tok_tables \
        --weights "$CK" --n 200 --temp 0.8 --seed 11 --attn blas 2>/dev/null > "$OUT/sample.txt"
    local TS=$(ls /tmp/aq/build_*/app/texture_scan 2>/dev/null | tail -1)   # o scan vive no build do proprio braco
    [ -n "$TS" ] && TX=$("$TS" "$OUT/sample.txt" --json 2>/dev/null | $PY -c "import json,sys; print(json.load(sys.stdin)['texture']['palavra_plausivel'])" 2>/dev/null || echo n/d)
  fi
  printf "%-16s %-9s %-11s %-9s %-11s %-8s\n" "$TAG" "$HD" "$KVB" "$SP" "${B:-n/d}" "$TX"
}

# baseline and the MQA control, then head_dim alone; seed 7 for all
run_arm 6 2 7
run_arm 6 1 7
run_arm 3 1 7
run_arm 2 1 7
# second seed on the two decisive arms
run_arm 6 1 13
run_arm 2 1 13

echo
echo "# Read: compare (6,1,16) against (3,1,32) and (2,1,48) for the head_dim"
echo "# effect at fixed kv. Compare (6,2,16) against (6,1,16) for the MQA effect"
echo "# at fixed head_dim. The FLOPs are equal in every arm."
echo "fim archqual $(date '+%F %H:%M:%S')"
