#!/bin/bash
#SBATCH -J m25b
#SBATCH -p compute
#SBATCH -c 10
#SBATCH --mem 32G
#SBATCH -t 72:00:00
#SBATCH -o /home/iangiestas/autoresearch-eval/m25b_%j.log
# M25 branch B (halfbeast): MESMA init seed 21 + janela [2000, 61889).
# Par A no fermi: mesma init, janela [0, 59889). Sopa iff drift<0.1.
set -u
EV=$HOME/autoresearch-eval
echo "m25b job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 OMP_DYNAMIC=FALSE \
  "$EV/portable/run_train_run.sh" --weights "$EV/m25_init" --rows "$EV/rows_m25b.npy" \
    --out /tmp/w_m25b --nsteps 59889 --lr 0.0003 --start_row 2000 --ntrain 61889 \
    --nval 5 --val_every 1000 --save_every 4000 --keep_last 2 --trn_probe 200 \
    --attn blas --bytes "$EV/tok_tables/token_bytes.txt"
echo "fim m25b $(date '+%F %H:%M:%S')"
