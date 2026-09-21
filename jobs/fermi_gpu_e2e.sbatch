#!/bin/bash
#SBATCH -J gpue2e
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o /home/pauli/autoresearch/logs/gpue2e_%j.log
# End-to-end no formato do C-0.1B (d768), onde a GPU ganha.
#
# Por que neste formato e nao no d96: medido, a GPU faz 1154 a 1788 GFLOP/s nas
# formas do 88M e apenas 126 a 181 nas formas do d96. Nas formas pequenas o custo
# fixo de transferencia e sincronizacao por chamada domina. Uma corrida em d96
# mediria o overhead, nao a GPU.
#
# O numero que conta e' o tokens por segundo, contra os 566 da CPU medidos na
# sonda do C-0.1B (job 165, que roda nesta mesma maquina: 8 threads dele, 8
# aqui, e a contencao fica declarada).
#
# O loss tambem sai, mas com uma ressalva ja' medida: rocBLAS e OpenBLAS somam em
# ordens diferentes, entao os dois caminhos divergem por amplificacao caotica de
# roundoff de float32. A conferencia justa e' a qualidade estatistica em passos
# iguais, nao o ultimo digito.
set -u
cd /home/pauli/autoresearch
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
INIT=/tmp/c01b/init                 # a init do 88M, do job 165. So' LEITURA.
ROWS=/tmp/mix/rows_f0.npy
N=100
bin/env_record "inicio do e2e gpu (fermi, 165 rodando)"

echo "=== 1) build CPU d768 (caminho declarativo ou fallback)"
bin/build_arch 768 6 2 12 8192 1024 /tmp/z768_cpu 2>&1 | tail -2
CPU=$(ls /tmp/z768_cpu/*/app/train_run 2>/dev/null | head -1)
[ -n "$CPU" ] || { echo "SEM BINARIO CPU"; exit 2; }

echo "=== 2) build GPU d768"
nix develop .#rocm --command bin/build_gpu 768 6 2 12 8192 1024 /tmp/z768_gpu 2>&1 | tail -2
GPU=$(ls /tmp/z768_gpu/*/app/train_run 2>/dev/null | head -1)
[ -n "$GPU" ] || { echo "SEM BINARIO GPU"; exit 2; }
echo "cpu=$CPU"; echo "gpu=$GPU"

for M in cpu gpu; do
  B=$CPU; [ "$M" = gpu ] && B=$GPU
  echo "########## $M"
  bin/env_record "antes de $M"
  T0=$(date +%s.%N)
  AUTORESEARCH_BLAS=$M OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    $B --weights $INIT --rows $ROWS --out /tmp/z768_$M \
    --nsteps $N --lr 6e-4 --ntrain 11797 --start_row 0 \
    --attn blas --batch 1 --bytes $BY 2>&1 | grep -E "^step (1|10|50|100) |FATAL" | head -6
  T1=$(date +%s.%N)
  awk -v t0=$T0 -v t1=$T1 -v n=$N -v m=$M 'BEGIN{
    s=t1-t0; printf "  %s: %.1f s para %d passos = %.3f s/passo = %.0f tok/s (1024 tok/passo)\n", m, s, n, s/n, n*1024/s }'
done
echo "fim gpue2e $(date '+%F %H:%M:%S')"
