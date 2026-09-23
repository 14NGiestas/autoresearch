#!/bin/bash
#SBATCH -J qa2
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -t 02:00:00
#SBATCH -o /home/iangiestas/autoresearch-eval/qa2_%j.log
# Formato DIVERSO (boolq+ARC, 10.5k pares, 0.9 epocas): formato transfere?
set -u
EV=$HOME/autoresearch-eval
echo "qa2 job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
  "$EV/portable/lib/ld-linux-x86-64.so.2" --library-path "$EV/portable/lib" "$EV/ov_train_run" \
  --weights "$EV/f0_best" --rows "$EV/rows_qa2.npy" --out /tmp/w_qa2 --nsteps 2000 --lr 0.0003 \
  --ntrain 2000 --nval 5 --val_every 500 --save_every 2000 --keep_last 1 --trn_probe 200 \
  --attn blas --bytes "$EV/tok_tables/token_bytes.txt"
echo "fim qa2 $(date '+%F %H:%M:%S')"
