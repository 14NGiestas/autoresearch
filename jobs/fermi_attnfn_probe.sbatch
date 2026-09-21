#!/bin/bash
#SBATCH -J attnfn
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /home/pauli/autoresearch/logs/attnfn_%j.log
# Os tres modos de atencao, no mesmo protocolo de 900 passos.
#
# Por que: o relu/T foi medido e ficou pior (gap crescente, +0,156 bpb em 2300
# passos). A literatura aponta a normalizacao L1, e nao a exponencial, como o
# componente critico do softmax, e a variante relu/sum(relu) era a unica que
# nunca foi testada. Agora esta' provada no portao (FD 1e-5 nos tres) e e'
# alcancavel do modelo.
#
# Os tres bracos no mesmo job, na mesma maquina, com a mesma init e as mesmas
# linhas: so' assim a comparacao tem valor. O /T e o softmax ja' foram medidos no
# halfbeast; aqui repetem-se para o contraste ser interno.
set -u
cd /home/pauli/autoresearch
BY=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
D=/tmp/mix
BD=/tmp/attnfn_bd/arch_d96_h6_kv2_l12_v8192_c1024
bin/job_env 96 6 2 12 8192 1024 "$BD" \
  "relu_l1:src/lib/fortran_train.f90" "attn-fn:src/app/train_run.f90" || exit 2
. "$BD/job_env.sh"
echo "fonte: $SRC_HASH  binario: $TRAIN_RUN"
# a val precisa do holdout concatenado: o train_run pontua [ntrain, ntrain+nval)
if [ ! -f $D/rows_f0h.npy ]; then
  .venv-numpy/bin/python3 -c "
import numpy as np
a=np.load('$D/rows_f0.npy'); b=np.load('$D/rows_holdout.npy')
np.save('$D/rows_f0h.npy', np.concatenate([a,b],axis=0)); print('concat', a.shape, b.shape)"
fi
for FN in softmax relu relu_l1; do
  echo "########## braco $FN"
  bin/env_record "antes de $FN"
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
    "$TRAIN_RUN" --weights $D/init3m --rows $D/rows_f0h.npy --out /tmp/attnfn/$FN \
    --nsteps 900 --lr 6e-4 --ntrain 11797 --start_row 0 \
    --nval 100 --val_every 100 --trn_probe 200 --save_every 900 --keep_last 1 \
    --attn blas --batch 1 --bytes $BY --attn-fn $FN
done
echo "fim attnfn $(date '+%F %H:%M:%S')"
