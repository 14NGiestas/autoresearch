#!/bin/bash
#SBATCH -J evalbench
#SBATCH -p compute
#SBATCH -c 10
#SBATCH --mem 24G
#SBATCH -t 01:00:00
#SBATCH -o /home/iangiestas/autoresearch-eval/evalbench_%j.log
# Benchmark eval_bpb --batch 1 vs 8 (120 rows, caixa quieta, OMP=10).
set -u
EV=$HOME/autoresearch-eval
W=/tmp/w_p2b2/step_7812
ls "$W/arch.txt" > /dev/null || { echo "pesos B2 sumiram de /tmp"; exit 1; }
head -120 "$EV/val_f0.txt" > /tmp/val120.txt
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 OMP_DYNAMIC=FALSE
echo "== batch 1"; time "$EV/portable/run_eval_bpb.sh" \
  --weights "$W" --rows /tmp/val120.txt --batch 1 > /tmp/v120_b1.txt 2>/tmp/v120_b1.err
echo "== batch 8"; time "$EV/portable/run_eval_bpb.sh" \
  --weights "$W" --rows /tmp/val120.txt --batch 8 > /tmp/v120_b8.txt 2>/tmp/v120_b8.err
cmp -s /tmp/v120_b1.txt /tmp/v120_b8.txt && echo "OUTPUT IDENTICO" || echo "OUTPUT DIFERE"
echo "fim evalbench $(date +%H:%M:%S)"
