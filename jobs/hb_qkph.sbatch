#!/bin/bash
#SBATCH -J qkph01
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -t 24:00:00
#SBATCH -o /home/iangiestas/autoresearch-eval/qkph01_%j.log
# QK-hop per-head (preserva especializacao): mesmos rows/init/steps/LR do f0.
set -u
EV=$HOME/autoresearch-eval
echo "qkph01 job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
  "$EV/portable/lib/ld-linux-x86-64.so.2" --library-path "$EV/portable/lib" "$EV/qk_train_run" \
  --weights "$HOME/init3m_qk" --rows "$EV/rows_f0.npy" --out /tmp/w_qkph \
  --nsteps 7812 --lr 0.0006 --ntrain 7812 --nval 5 --val_every 1000 \
  --save_every 4000 --keep_last 2 --trn_probe 200 --attn qkhop-ph \
  --bytes "$EV/tok_tables/token_bytes.txt"
echo "fim qkph01 $(date '+%F %H:%M:%S')"
