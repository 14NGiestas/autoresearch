#!/bin/bash
#SBATCH -J epochs
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o /home/pauli/autoresearch/logs/epochs_%j.log
# hyp_f48678: o deficit de K shards fundidos e falta de EPOCAS ou de TOKENS
# DISTINTOS? Aqui: K=4 com 2 epocas da propria fatia (3904 passos) + merge,
# contra (a) run unico com o MESMO compute total (15616 passos, dados completos)
# e (b) run unico com o MESMO wall-clock (3904 passos).
# 1 epoca ja temos: shards 2.440-2.457, merge one-shot 2.504, run unico 1952 = 2.44475.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
OUT=/tmp/ep2; mkdir -p $OUT
echo "epochs job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
tr() { # $1=out $2=start_row $3=ntrain $4=nsteps
  [ -d "$1/step_$4" ] && return 0
  rm -rf "$1"
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
    --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out "$1" --nsteps "$4" \
    --lr 6e-4 --ntrain "$3" --start_row "$2" --nval 1 --val_every 9999999 \
    --save_every "$4" --trn_probe 1 --attn blas --bytes "$BY" > "$1.log" 2>&1
}
ev() { OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
    --weights "$1" --rows /tmp/mix/rows_holdout.npy --attn blas --batch 16 > "$2" 2>&1; }
for k in 0 1 2 3; do tr $OUT/s$k $((k*1953)) 1953 3904; done
PY=.venv-numpy/bin/python3
$PY scripts/merge_checkpoints.py $OUT/s0/step_3904 $OUT/s1/step_3904 $OUT/m01 --alpha 0.5 >/dev/null 2>&1
$PY scripts/merge_checkpoints.py $OUT/m01 $OUT/s2/step_3904 $OUT/m012 --alpha 0.6666667 >/dev/null 2>&1
$PY scripts/merge_checkpoints.py $OUT/m012 $OUT/s3/step_3904 $OUT/merge --alpha 0.75 >/dev/null 2>&1
tr $OUT/base3904 0 7812 3904
tr $OUT/base15616 0 7812 15616
for d in s0 s1 s2 s3; do ev $OUT/$d/step_3904 $OUT/ev_$d.txt; done
ev $OUT/merge $OUT/ev_merge.txt
ev $OUT/base3904/step_3904 $OUT/ev_base3904.txt
ev $OUT/base15616/step_15616 $OUT/ev_base15616.txt
echo "fim epochs $(date '+%F %H:%M:%S')"
