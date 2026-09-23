#!/bin/bash
#SBATCH -J attnfnlong
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o /home/pauli/autoresearch/logs/attnfnlong_%j.log
# A L1 a 2300 passos: o gap cresce ou estabiliza?
#
# A sonda de 900 passos (job 174) deu: softmax 2,70798, relu/T 2,71965,
# relu/sum 2,71443. A L1 melhora o /T por 0,005 e fica 0,0065 atras do softmax.
#
# Mas o que se sabe do /T e' que o gap com o softmax CRESCE com o comprimento:
# -0,01 nos primeiros 500 passos, +0,131 em 1900, +0,156 em 2300. Se o gap da L1
# tambem cresce, o resultado de hoje e' definitivo; se estabiliza, a L1 pode
# alcancar o softmax em corridas longas, e isso muda a conclusao.
#
# Dois bracos aqui, nao tres: o /T ja' esta' medido ate' 2300 passos.
set -u
cd /home/pauli/autoresearch
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
D=/tmp/mix
BD=/tmp/attnfnlong_bd/arch_d96_h6_kv2_l12_v8192_c1024
bin/job_env 96 6 2 12 8192 1024 "$BD" \
  "relu_l1:src/lib/fortran_train.f90" "attn-fn:src/app/train_run.f90" || exit 2
. "$BD/job_env.sh"
echo "fonte: $SRC_HASH  binario: $TRAIN_RUN"
[ -f $D/rows_f0h.npy ] || .venv-numpy/bin/python3 -c "
import numpy as np
a=np.load('$D/rows_f0.npy'); b=np.load('$D/rows_holdout.npy')
np.save('$D/rows_f0h.npy', np.concatenate([a,b],axis=0))"
for FN in softmax relu_l1; do
  echo "########## braco $FN"
  bin/env_record "antes de $FN"
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    "$TRAIN_RUN" --weights $D/init3m --rows $D/rows_f0h.npy --out /tmp/attnfnlong/$FN \
    --nsteps 2300 --lr 6e-4 --ntrain 11797 --start_row 0 \
    --nval 100 --val_every 100 --trn_probe 200 --save_every 2300 --keep_last 1 \
    --attn blas --batch 1 --bytes $BY --attn-fn $FN
done
echo "fim attnfnlong $(date '+%F %H:%M:%S')"
