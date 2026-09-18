#!/bin/bash
#SBATCH -J leak
#SBATCH -c 8
#SBATCH -t 01:30:00
#SBATCH -o /home/pauli/autoresearch/logs/leak_%j.log
# Hipotese: o trainer vaza memoria a cada checkpoint (~55 MB/save). Evidencia: o
# job 103 chegou a 3.5 GB de RSS depois de 62 saves e morreu em OOM; runs curtos
# (poucos saves) nunca morreram. Aqui: 2000 passos com save_every 100 (20 saves)
# imprimindo RSS a cada save, em dois modos (npy e st) para isolar o culpado.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
rss() { awk '/VmRSS/{print $2/1024" MB"}' /proc/$1/status 2>/dev/null; }
for MODE in npy st; do
  rm -rf /tmp/leak_$MODE
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
    --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/leak_$MODE \
    --nsteps 2000 --lr 6e-4 --ntrain 7812 --nval 1 --val_every 9999999 \
    --save_every 100 --ckpt-format $MODE --trn_probe 1 --attn blas --bytes "$BY" \
    > /tmp/leak_$MODE.log 2>&1 &
  PID=$!
  echo "--- modo $MODE (pid $PID)"
  while kill -0 $PID 2>/dev/null; do
    S=$(grep -cE "^step " /tmp/leak_$MODE.log 2>/dev/null || echo 0)
    echo "   passo $S -> RSS $(rss $PID)"
    sleep 45
  done
  wait $PID; echo "   fim $MODE: RSS final n/a"
done
echo "fim leak $(date '+%F %H:%M:%S')"
