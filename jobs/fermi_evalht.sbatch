#!/bin/bash
#SBATCH -J evalht
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /home/pauli/autoresearch/logs/evalht_%j.log
# Benchmark HT no eval: OMP 4/8/16 em 50 rows (timing puro).
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
for O in 4 8 16; do
  echo "== OMP=$O"
  OMP_NUM_THREADS=$O OPENBLAS_NUM_THREADS=$O OMP_DYNAMIC=FALSE /usr/bin/time -f "elapsed %es" "$BIN" \
    --weights /tmp/mix/w_f0/best --rows /tmp/mix/bench50.npy --attn blas --batch 8 \
    > /dev/null 2>/tmp/ht_$O.err; tail -n 1 /tmp/ht_$O.err
done
echo "fim evalht $(date '+%F %H:%M:%S')"
