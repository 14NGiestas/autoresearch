#!/bin/bash
#SBATCH -J reludecay
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem 8G
#SBATCH -t 02:00:00
#SBATCH -o /tmp/evalfix_pauli_20260920/reludecay_%j.log
# Piso diferente, ou colapso? A CURVA responde, o valor final nao.
#
# O job 164 deu relu 4,154340 contra softmax 2,352489. Se o relu cai na mesma
# inclinacao e para mais alto, e' piso diferente (a atencao funciona). Se achata
# cedo, e' colapso (a atencao morreu). A curva do 164 perdeu-se porque o job
# passava o log por tail -6. Aqui o log vai inteiro, val a cada 100 passos.
#
# Todo o preparo sai do bin/job_env: preflight da fonte, hash da fonte, build com
# fpm e o caminho publicado. Este job NAO tem glob com tail -1, que foi o que
# quebrou o 155 e que eu repeti aqui.
#
# Roda no halfbeast porque a fermi esta' com a sonda do C-0.1B (job 165).
set -u
D=/tmp/evalfix_pauli_20260920
cd "$D" || exit 1
BD="$D/bd/arch_d96_h6_kv2_l12_v8192_c1024"
bin/job_env 96 6 2 12 8192 1024 "$BD" \
  "attn-fn:src/app/train_run.f90" "relu_attn:src/lib/fortran_attn.f90" \
  "pos_frac:src/app/eval_bpb.f90" || exit 2
. "$BD/job_env.sh"
echo "fonte: SRC_HASH=$SRC_HASH  binario: $TRAIN_RUN"
BY="$D/tok_tables/token_bytes.txt"
for FN in softmax relu; do
  echo "--- braco $FN"
  bin/env_record "antes de $FN"
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_DYNAMIC=FALSE \
    "$TRAIN_RUN" --weights $D/init3m --rows $D/rows_f0.npy --holdout $D/rows_holdout.npy \
    --out /tmp/decay/$FN --nsteps 900 --lr 6e-4 --ntrain 11797 --start_row 0 \
    --nval 5 --val_every 100 --trn_probe 200 --save_every 900 --keep_last 1 \
    --attn blas --batch 1 --bytes "$BY" --attn-fn $FN
done
echo "fim reludecay $(date '+%F %H:%M:%S')"
