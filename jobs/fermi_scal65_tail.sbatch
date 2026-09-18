#!/bin/bash
#SBATCH -J scal65tail
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /home/pauli/autoresearch/logs/scal65tail_%j.log
# Repoe os DOIS ultimos pontos da curva do job 103, que morreram em OOM quando o
# smoke test MPI (train_ddp, fora de sbatch) estourou os 15 GB da fermi.
# Continuacao EXATA: LR constante + --start_row 62464 --ntrain 63889 fazem o
# stream de dados continuar de onde parou; --t0 62465 da os mesmos nomes de dir.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
echo "scal65tail job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
  --weights /tmp/scal65/step_62464 --rows /tmp/prose65/rows_train.npy --out /tmp/scal65_tail \
  --nsteps 1425 --t0 62465 --lr 6e-4 --ntrain 63889 --start_row 62464 --nval 1 \
  --val_every 9999999 --save_every 1024 --trn_probe 1 --attn blas --bytes "$BY" \
  > /tmp/scal65_tail.log 2>&1
for d in /tmp/scal65_tail/step_*; do
  s=$(basename $d | sed 's/step_//')
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
    --weights "$d" --rows /tmp/prose65/rows_val.npy --attn blas --batch 16 \
    > /tmp/scal65/ev_$s.txt 2>&1
  echo "  ev_$s ok"
done
echo "fim scal65tail $(date '+%F %H:%M:%S')"
