#!/bin/bash
#SBATCH -J m25a
#SBATCH -c 8
#SBATCH -t 72:00:00
#SBATCH -o /home/pauli/autoresearch/logs/m25a_%j.log
# M25 branch A (fermi): init seed 21 compartilhada + janela [0, 59889).
# Par B no halfbeast: mesma init, janela [2000, 61889) (receita v2/v3).
# Sopa iff drift<0.1 (hyp_b6e33a).
set -u
cd /home/pauli/autoresearch
E=$(ls -t src/build/arch_d360_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE
"$E" --weights /tmp/m25_init --rows /tmp/mix/rows_m25b.npy --out /tmp/w_m25a \
  --nsteps 59889 --lr 0.0003 --start_row 0 --ntrain 61889 --nval 5 \
  --val_every 1000 --save_every 4000 --keep_last 2 --trn_probe 200 --attn blas \
  --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
echo "fim m25a $(date '+%F %H:%M:%S')"
