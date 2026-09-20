#!/bin/bash
#SBATCH -J archsota
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o /home/pauli/autoresearch/logs/archsota_%j.log
# Does the SOTA-like shape run at our scale, and how fast is the step?
#
# The field uses head_dim 128 (Llama 3: 32 to 128 heads, 8 kv heads, head_dim
# 128; Gemma 2 uses 256). Our d96 has head_dim 16, which is 8 to 16 times
# narrower, and that is the measured cause of the attention cost. A head_dim of
# 128 does not fit in d_model 96 with more than one head, so the model must grow.
#
# This job builds two SOTA-like shapes at d_model 256 and trains each for a short
# run. It measures the seconds for one step and watches the loss for a
# divergence, because the field reports that wide heads need a stabilizer such as
# QK-Norm or logit soft-capping. The measurement comes first: a stabilizer is
# only worth a new kernel if this run shows a problem.
set -u
cd /home/pauli/autoresearch
export OMP_NUM_THREADS=8
PY=.venv-numpy/bin/python3
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
ROWS=/tmp/mix/rows_f0.npy
STEPS=300
echo "archsota job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"
printf "%-14s %-7s %-6s %-9s %-10s %-11s %-10s\n" "arm" "d_model" "heads" "kv" "head_dim" "s/step" "nll fim"

arm () {  # d heads kv
  local D=$1 H=$2 KV=$3
  local HD=$((D/H))
  local TAG="d${D}_h${H}_kv${KV}"
  local OUT=/tmp/sota/$TAG BD=/tmp/sota/build_$TAG
  rm -rf "$OUT"
  $PY scripts/make_init.py --out "$OUT" --d "$D" --heads "$H" --kv "$KV" --layers 12 \
      --vocab 8192 --ctx 1024 --seed 7 >/dev/null || { echo "  $TAG: init falhou"; return; }
  if [ -z "$(ls $BD/*/app/train_run 2>/dev/null | tail -1)" ]; then
    bin/build_arch "$D" "$H" "$KV" 12 8192 1024 "$BD" > "$BD.log" 2>&1 || { echo "  $TAG: build falhou"; tail -2 "$BD.log"; return; }
  fi
  local BIN=$(ls $BD/*/app/train_run | tail -1)
  local T0=$(date +%s.%N)
  $BIN --weights "$OUT" --rows $ROWS --out "$OUT/run" --nsteps $STEPS --lr 6e-4 \
      --ntrain 11797 --start_row 0 --nval 1 --val_every 9999999 --trn_probe 1 \
      --save_every $STEPS --attn blas --batch 1 --bytes "$BY" > "$OUT/train.log" 2>&1
  local T1=$(date +%s.%N)
  local SP=$(echo "($T1-$T0)/$STEPS" | bc -l | cut -c1-5)
  local L=$(grep '^step' "$OUT/train.log" | tail -1 | awk '{print $4}')
  local L0=$(grep '^step' "$OUT/train.log" | head -1 | awk '{print $4}')
  printf "%-14s %-7s %-6s %-9s %-10s %-11s %-10s\n" "$TAG" "$D" "$H" "$KV" "$HD" "$SP" "${L:-?}"
  echo "      (nll passo 1 = ${L0:-?}; divergencia = nll subindo ou NaN)"
}
arm 256 2 1     # 2 heads x 128, MQA
arm 256 4 2     # 4 heads x 64, GQA (a razao do Llama 2)
arm 96 2 1      # controle: a forma de hoje com 2 cabecas largas de 48
echo "fim archsota $(date '+%F %H:%M:%S')"
