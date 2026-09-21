#!/bin/bash
#SBATCH -J reluarm
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem 8G
#SBATCH -t 02:00:00
#SBATCH -o /tmp/evalfix_pauli_20260920/reluarm_%j.log
# Um braco so'. Uso: sbatch --export=FN=relu halfbeast_relu_arm.sbatch
#
# Por que um braco por job: os dois bracos num job so' bateram no limite de parede
# de 2 h, e o job morreu com 2500 de 2929 passos do PRIMEIRO braco. O segundo
# nunca comecou. Dois jobs de 4 threads rodam em paralelo e ficam prontos em ~50
# min, em vez de 100.
#
# A pergunta: o job 164 deu relu 4,154340 contra softmax 2,352489, com as mesmas
# flags. A sonda de 900 passos mostrou as duas curvas identicas, com diferenca
# dentro de +-0.013. Entao a diferenca do 164, se existe, aparece entre 900 e 2929.
#
# rows_f0h.npy e' treino mais holdout concatenados (11897 linhas). O train_run nao
# tem --holdout: ele pontua as linhas [start_row+ntrain, +nval) do PROPRIO arquivo.
# Com nval=100 o val pontua exatamente o holdout, os mesmos 100 que o driver usa,
# e por isso os numeros ficam comparaveis com o 164.
set -u
D=/tmp/evalfix_pauli_20260920
cd "$D" || exit 1
FN=${FN:?defina FN=softmax ou FN=relu}
BD="$D/bd/arch_d96_h6_kv2_l12_v8192_c1024"
bin/job_env --build-dir "$BD" "attn-fn:src/app/train_run.f90" \
  "relu_attn:src/lib/fortran_attn.f90" || exit 2
. "$BD/job_env.sh"
echo "braco=$FN  SRC_HASH=$SRC_HASH"
bin/env_record "antes de $FN"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_DYNAMIC=FALSE \
  "$TRAIN_RUN" --weights $D/init3m --rows $D/rows_f0h.npy \
  --out /tmp/decay29/$FN --nsteps 2929 --lr 6e-4 --ntrain 11797 --start_row 0 \
  --nval 100 --val_every 100 --trn_probe 200 --save_every 2929 --keep_last 1 \
  --attn blas --batch 1 --bytes "$D/tok_tables/token_bytes.txt" --attn-fn $FN
echo "fim braco $FN $(date '+%F %H:%M:%S')"
