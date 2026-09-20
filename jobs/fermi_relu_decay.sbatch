#!/bin/bash
#SBATCH -J reludecay
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/reludecay_%j.log
# A pergunta: o relu ficou PIOR, ou e' so' um PISO diferente?
#
# O job 164 deu relu 4,154340 contra softmax 2,352489. Duas leituras possiveis, e
# elas sao distinguiveis pela CURVA, nao pelo valor final:
#   (a) piso diferente: o bpb cai na mesma inclinacao e para mais alto. A atencao
#       funciona, a normalizacao e' que custa qualidade.
#   (b) colapso: o bpb achata cedo. A atencao morreu (difusa ou com cabecas
#       mortas), que e' o risco que o proprio survey avisou.
#
# A curva do 164 PERDEU-SE: o job passava o log por `tail -6`, entao so' as
# ultimas linhas sobreviveram. Aqui o log vai inteiro, e o val e' a cada 100
# passos para dar pontos suficientes para ajustar E + A*D^-beta (o scripts/gap_law.py).
#
# 900 passos x ~0,8 s = ~12 min por braco. 4 threads para coexistir com a sonda do
# C-0.1B (job 165). O bpb e' deterministico, entao contencao nao o afeta.
set -u
cd /home/pauli/autoresearch
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
B=$(ls /tmp/rl4/bd/*/gfortran_*/app/train_run | tail -1)
[ -n "$B" ] || { echo "sem binario"; exit 1; }
echo "reludecay job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
for FN in softmax relu; do
  echo "--- braco $FN"
  bin/env_record "antes de $FN"
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_DYNAMIC=FALSE \
    "$B" --weights /tmp/mix/init3m --rows /tmp/mix/rows_f0.npy \
    --out /tmp/decay/$FN --nsteps 900 --lr 6e-4 --ntrain 11797 --start_row 0 \
    --nval 5 --val_every 100 --trn_probe 200 --save_every 900 --keep_last 1 \
    --attn blas --batch 1 --bytes "$BY" --attn-fn $FN
done
echo "fim reludecay $(date '+%F %H:%M:%S')"
