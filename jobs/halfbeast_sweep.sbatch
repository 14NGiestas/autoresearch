#!/bin/bash
#SBATCH -J sweep3m_hb
#SBATCH --array=0-1
#SBATCH -c 5
#SBATCH --mem 24G
#SBATCH -t 04:00:00
#SBATCH -o /home/iangiestas/autoresearch-eval/sweep_hb_%A_%a.log
# Braços 3 (75% wiki) e 4 (100% wiki) da dose-resposta, despachados para a caixa
# remota enquanto o fermi termina os outros três. -c 5 porque o job 6010 (do
# iangiestas) segura 10 CPUs: educado, e cabe nos 10 threads físicos restantes.
# O binário é o bundle portable (compilado na fermi): mesma arquitetura de rede e
# mesma OpenBLAS, então o bpb é comparável -- só a CPU é diferente (e por isso a
# VELOCIDADE não é comparável, o que não importa para uma curva de qualidade).
set -u
EV=$HOME/autoresearch-eval
FS=(75 100)
f=${FS[$SLURM_ARRAY_TASK_ID]}
echo "arm wiki=${f}%  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  host=$(hostname)  $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 OMP_DYNAMIC=FALSE \
  "$EV/portable/run_train_run.sh" --weights "$EV/init3m" --rows "$EV/rows_f$f.txt" \
    --out /tmp/w_hb_f$f --nsteps 7812 --lr 0.0006 --ntrain 7812 --nval 5 \
    --val_every 1000 --save_every 4000 --keep_last 1 --trn_probe 200 --attn blas \
    --bytes "$EV/tok_tables/token_bytes.txt"
echo "fim arm ${f}%  $(date '+%H:%M:%S')"
