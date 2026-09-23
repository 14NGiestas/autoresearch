#!/bin/bash
#SBATCH -J relu3m
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /home/pauli/autoresearch/logs/relu3m_%j.log
# Softmax contra ReLU-attention, no testbed de 3M tokens.
#
# Por que: a medicao de hoje (bench_attn_split) mostrou que o softmax e' 71,6% da
# atencao forward, e o exp esta' tambem no elementwise do backward. Entao atacar o
# exp ataca as duas fases caras de uma vez. E' o unico item do survey de atencao
# que sobrou com argumento mecanico no nosso regime, e e' prova-de-escala (ao
# contrario do save-P, que precisa de 34 GB na escala do alvo).
#
# Dois bracos, MESMO init, MESMOS dados, MESMOS passos. A unica mudanca e' a
# funcao de atencao. Mede loss E textura (H2), porque bpb sozinho nao diz se a
# geracao presta -- foi o achado da sessao.
set -u
cd /home/pauli/autoresearch
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
BIN=/tmp/rl3/bd/arch_d96_h6_kv2_l12_v8192_c1024/gfortran_*/app/train_run
B=$(ls $BIN | tail -1)
[ -n "$B" ] || { echo "sem binario"; exit 1; }
echo "relu3m job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
bin/env_record "antes"
for FN in softmax relu; do
  echo "--- braco $FN"
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    "$B" --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy \
    --out /tmp/rl3m/$FN --nsteps 2929 --lr 6e-4 --ntrain 11797 --start_row 0 \
    --nval 5 --val_every 293 --trn_probe 200 --save_every 2929 --keep_last 1 \
    --attn blas --batch 1 --bytes "$BY" --attn-fn $FN 2>&1 | tail -4
done
bin/env_record "depois"
echo "fim relu3m $(date '+%F %H:%M:%S')"
