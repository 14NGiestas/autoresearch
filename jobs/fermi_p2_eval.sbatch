#!/bin/bash
#SBATCH -J p2eval
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/p2eval_%j.log
# P2 full-val evals (3985 rows, d216): branch B2 final + same-init soup v2.
# B2 shares initA with branch A (drift 0.518, same basin) -> soup should win.
set -u
cd /home/pauli/autoresearch
E=$(ls -t src/build/arch_d216_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE
echo "== B2 final $(date +%H:%M)"
"$E" --weights /tmp/w_p2b2_final --rows /tmp/mix/val_f0.txt \
  > /tmp/eval_p2b2.txt 2>/tmp/eval_p2b2.err
echo "== soup v2 (A+B2) $(date +%H:%M)"
"$E" --weights /tmp/w_p2soup2 --rows /tmp/mix/val_f0.txt \
  > /tmp/eval_p2soup2.txt 2>/tmp/eval_p2soup2.err
echo "fim p2eval $(date '+%F %H:%M:%S')"
