#!/bin/bash
#SBATCH -J arcqa
#SBATCH -c 4
#SBATCH --mem 8G
#SBATCH -t 01:00:00
#SBATCH -o /home/iangiestas/autoresearch-eval/arcqa_%j.log
set -u
EV=$HOME/autoresearch-eval
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_DYNAMIC=FALSE
"$EV/portable/lib/ld-linux-x86-64.so.2" --library-path "$EV/portable/lib" "$EV/ov_eval_bpb" \
  --weights /tmp/w_qa/best --rows "$EV/arc_rows.npy" --attn blas --batch 8 \
  > "$EV/arc_qa.txt" 2>&1
echo "fim arcqa $(date '+%F %H:%M:%S')"
