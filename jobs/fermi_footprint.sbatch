#!/bin/bash
#SBATCH -J footp
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -o /home/pauli/autoresearch/logs/footp_%j.log
# Os 4.4 GB fixos do d96 sao custo de PARAMETRO (escala mal: d216/d360 seriam
# piores) ou de CONTEXTO (T^2*heads*layers, igual em qualquer modelo: escala bem)?
# Resposta: roda o d216 (10M params, mesmo T=1024) por 120 passos e le o RSS.
set -u
cd /home/pauli/autoresearch
for A in d96 d216; do
  BIN=$(ls -t src/build/arch_$A*/gfortran_*/app/train_run 2>/dev/null | head -1)
  W=/tmp/mix/init3m; R=/tmp/mix/rows_f0.npy
  [ "$A" = d216 ] && { W=/tmp/mix200/init10m; R=/tmp/mix200/rows_mix200.npy; }
  [ -x "$BIN" ] || { echo "  $A: binario ausente"; continue; }
  rm -rf /tmp/footp_$A
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$BIN" \
    --weights $W --rows $R --out /tmp/footp_$A --nsteps 120 --lr 6e-4 --ntrain 64 \
    --nval 1 --val_every 9999999 --save_every 9999999 --trn_probe 1 --attn blas \
    --bytes "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt" > /tmp/footp_$A.log 2>&1 &
  PID=$!
  sleep 60
  echo "  $A: passo $(grep -cE '^step ' /tmp/footp_$A.log) RSS $(awk '/VmRSS/{print $2/1024" MB"}' /proc/$PID/status 2>/dev/null)"
  kill $PID 2>/dev/null; wait $PID 2>/dev/null || true
done
echo "fim footp $(date '+%F %H:%M:%S')"
