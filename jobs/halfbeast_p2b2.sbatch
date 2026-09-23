#!/bin/bash
#SBATCH -J p2_hb2
#SBATCH -c 10
#SBATCH --mem 24G
#SBATCH -t 24:00:00
#SBATCH -o /home/iangiestas/autoresearch-eval/p2_hb2_%j.log
# P2 branch B, tentativa 2: MESMO init da branch A (seed 7) + ordem diferente
# (rows_p2b2 = mix f25 bloco 32768 + cauda val_f0, layout igual ao sweep).
# Tentativa 1 morreu metodologicamente: inits diferentes -> bacinos
# independentes (drift 1.42~=raiz2), sopa invalida (HEP hyp_167043).
set -u
EV=$HOME/autoresearch-eval
echo "p2 branch B2 job=${SLURM_JOB_ID} host=$(hostname) $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 OMP_DYNAMIC=FALSE \
  "$EV/portable/run_train_run.sh" --weights "$EV/p2_initA" --rows "$EV/rows_p2b2.txt" \
    --out /tmp/w_p2b2 --nsteps 7812 --lr 0.0003 --ntrain 7812 --nval 5 \
    --val_every 1000 --save_every 4000 --keep_last 2 --trn_probe 200 --attn blas \
    --bytes "$EV/tok_tables/token_bytes.txt"
echo "fim p2 branch B2 $(date '+%F %H:%M:%S')"
