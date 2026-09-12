#!/bin/bash
#SBATCH -J sweep3m
#SBATCH --array=0-4
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/sweep3m_%A_%a.log
# Dose-resposta do mix: 5 braços de 3M params x 8M tokens, MESMO total de tokens,
# só a fração de wiki muda (0/25/50/75/100%), com a prosa preenchendo o resto.
# Régua: bpb em prosa literária held-out (as linhas de val empacotadas no fim de
# rows_f*). O braço f=0 é o controle "só prosa" e o f=100 o controle "só wiki".
#
# Sem --mem: a memória do nó não está configurada no slurm.conf desta máquina
# (sinfo reporta 1MB) e pedir --mem faria o job ser rejeitado ou mal agendado.
# Sem -p: usa a partição default (debug).
set -u
cd /home/pauli/autoresearch
FS=(0 25 50 75 100)
f=${FS[$SLURM_ARRAY_TASK_ID]}
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
N=$(wc -l < /tmp/mix/train_f$f.txt)
echo "arm wiki=${f}%  rows=$N  job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}  host=$(hostname)  $(date '+%F %H:%M:%S')"
echo "bin=$BIN"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/init3m --rows /tmp/mix/rows_f$f.txt --out /tmp/mix/w_f$f \
  --nsteps 7812 --lr 0.0006 --ntrain $N --nval 5 --val_every 1000 \
  --save_every 4000 --keep_last 1 --trn_probe 200 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
echo "fim arm ${f}%  $(date '+%H:%M:%S')"
