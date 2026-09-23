#!/bin/bash
#SBATCH -J ann65
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/ann65_%j.log
# O achatamento e de DADOS/CAPACIDADE ou de OTIMIZACAO (LR constante)? Continua
# do checkpoint final do job 103 por 2000 passos com LR em cosseno ate zero:
# se o bpb cair muito, o "piso" que medimos e do LR, nao do modelo.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
CK=$(ls -d /tmp/scal65/step_* | sed 's/.*step_//' | sort -n | tail -1)
echo "ann65 job=${SLURM_JOB_ID} partindo de step_$CK $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/scal65/step_$CK --rows /tmp/prose65/rows_train.npy --out /tmp/ann65 \
  --nsteps 2000 --lr 6e-4 --anneal 1 --ntrain 63889 --start_row 0 --nval 1 \
  --val_every 9999999 --save_every 2000 --trn_probe 1 --attn blas --bytes "$BY" \
  > /tmp/ann65.log 2>&1
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
  --weights /tmp/ann65/step_2000 --rows /tmp/prose65/rows_val.npy --attn blas --batch 16 \
  > /tmp/ann65_ev.txt 2>&1
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
  --weights /tmp/scal65/step_$CK --rows /tmp/prose65/rows_val.npy --attn blas --batch 16 \
  > /tmp/ann65_ev_before.txt 2>&1
echo "fim ann65 $(date '+%F %H:%M:%S')"
