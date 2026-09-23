#!/bin/bash
#SBATCH -J scaling
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -o /home/pauli/autoresearch/logs/scaling_%j.log
# Curva de escala com PISO MEDIDO, nao com piso livre: UM run longo com
# --save_every 1024 entrega os pontos todos (LR constante => o checkpoint do
# passo N e identico a um run separado de N passos, entao a curva sai do mesmo
# trajeto). 32768 passos = 32M tokens; 32 pontos (1M..32M).
# Checagem embutida: o checkpoint do passo 3904 tem de bater com o run base3904
# ja medido (2.26189) -- se nao bater, a premissa "LR constante" esta errada.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
OUT=/tmp/scal; rm -rf $OUT; mkdir -p $OUT
echo "scaling job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out $OUT --nsteps 32768 \
  --lr 6e-4 --ntrain 7812 --start_row 0 --nval 1 --val_every 9999999 \
  --save_every 1024 --keep_last 40 --trn_probe 1 --attn blas --bytes "$BY" \
  > $OUT/train.log 2>&1
echo "treino ok $(date '+%H:%M:%S')"
for d in $OUT/step_*; do
  s=$(basename $d | sed 's/step_//')
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
    --weights "$d" --rows /tmp/mix/rows_holdout.npy --attn blas --batch 16 \
    > $OUT/ev_$s.txt 2>&1
  echo "  passo $s ($((s*1024/1000000))M tokens) ok"
done
echo "fim scaling $(date '+%F %H:%M:%S')"
