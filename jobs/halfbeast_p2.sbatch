#!/bin/bash
#SBATCH -J p2_hb
#SBATCH -c 10
#SBATCH --mem 24G
#SBATCH -t 24:00:00
#SBATCH -o /home/iangiestas/autoresearch-eval/p2_hb_%j.log
# P2 branch B (10M d216, dieta f=25, init seed 8, ordem bloco 32768).
# Gemea da branch A (fermi, init seed 7, rows_f25): mesma dieta,
# init/ordem diferentes -> sopa no fim. Fila vazia na submissao (-c 10 ok).
set -u
EV=$HOME/autoresearch-eval
echo "p2 branch B job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 OMP_DYNAMIC=FALSE \
  "$EV/portable/run_train_run.sh" --weights "$EV/p2_initB" --rows "$EV/rows_p2b.txt" \
    --out /tmp/w_p2b --nsteps 7812 --lr 0.0003 --ntrain 7812 --nval 5 \
    --val_every 1000 --save_every 4000 --keep_last 2 --trn_probe 200 --attn blas \
    --bytes "$EV/tok_tables/token_bytes.txt"
echo "fim p2 branch B $(date '+%F %H:%M:%S')"
