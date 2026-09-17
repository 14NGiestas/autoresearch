#!/bin/bash
#SBATCH -J scal65
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -o /home/pauli/autoresearch/logs/scal65_%j.log
# Curva de escala LIMPA (sem repeticao de pool): pool de prosa de 65,4M tokens
# frescos (rows_m25b[0:63889]) e holdout NOVO de 300 linhas (rows_m25b[63900:64200])
# que o treino nao contem. UM run com --save_every 1024 da 62 pontos de 1M a 65M
# tokens (LR constante => checkpoint do passo N == run de N passos).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
OUT=/tmp/scal65; rm -rf $OUT; mkdir -p $OUT
echo "scal65 job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/init3m --rows /tmp/prose65/rows_train.npy --out $OUT \
  --nsteps 63889 --lr 6e-4 --ntrain 63889 --start_row 0 --nval 1 \
  --val_every 9999999 --save_every 1024 --keep_last 70 --trn_probe 1 --attn blas \
  --bytes "$BY" > $OUT/train.log 2>&1
echo "treino ok $(date '+%H:%M:%S')"
for d in $OUT/step_*; do
  s=$(basename $d | sed 's/step_//')
  [ $((s % 2048)) -ne 0 ] && [ "$s" != "63889" ] && continue
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
    --weights "$d" --rows /tmp/prose65/rows_val.npy --attn blas --batch 16 \
    > $OUT/ev_$s.txt 2>&1
done
echo "evals ok, fim scal65 $(date '+%F %H:%M:%S')"
