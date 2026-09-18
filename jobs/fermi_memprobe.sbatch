#!/bin/bash
#SBATCH -J memprobe
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /home/pauli/autoresearch/logs/memprobe_%j.log
# O job 110 mostrou 4.403 MB de RSS JA nos primeiros ~100 passos e PLANO depois
# (400x o modelo). Nao e vazamento por passo: e uma alocacao fixa enorme. Aqui a
# gente separa anonimo (heap) de file-backed (mmap) e compara blas x naive.
set -u
cd /home/pauli/autoresearch
BIN=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
probe() { awk -v t="$2" '/VmRSS|VmPeak/{printf "%s %s ", $1, $2} /RssAnon|RssFile|RssShmem/{printf "%s=%s%s ", $1, $2, $3} END{print "| t="t"s"}' /proc/$1/smaps_rollup 2>/dev/null || awk '/VmRSS|VmPeak/{printf "%s=%s ", $1,$2} END{print ""}' /proc/$1/status; }
for MODE in blas naive; do
  rm -rf /tmp/memprobe_$MODE
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
    --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy --out /tmp/memprobe_$MODE \
    --nsteps 300 --lr 6e-4 --ntrain 7812 --nval 1 --val_every 9999999 \
    --save_every 250 --attn $MODE --trn_probe 1 --bytes "$BY" > /tmp/memprobe_$MODE.log 2>&1 &
  PID=$!
  echo "--- modo $MODE (pid $PID)"
  for w in 3 10 25; do sleep $w; S=$(grep -cE "^step " /tmp/memprobe_$MODE.log 2>/dev/null || echo 0); echo "   passo~$S $(probe $PID $w)"; done
  kill -0 $PID 2>/dev/null && kill $PID; wait $PID 2>/dev/null || true
done
echo "fim memprobe $(date '+%F %H:%M:%S')"
