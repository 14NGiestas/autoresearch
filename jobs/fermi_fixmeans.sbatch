#!/bin/bash
#SBATCH -J fixmean
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/fixmean_%j.log
# Recomputa as medias UNIFORMES EXATAS dos bracos afetados pelo bug de alpha do
# compose_grid.py (o "merge" antigo era [1/24,1/24,4/24,18/24] em K=4 e 0.875 no
# ultimo shard em K=8). Nao treina nada: so soma/N sobre checkpoints existentes.
set -u
cd /home/pauli/autoresearch
PY=.venv-numpy/bin/python3
EV=$(ls -t src/build/arch_d96_h6_kv2_l12_v8192_c1024/*/app/eval_bpb | head -1)
DIR=/tmp/fix; mkdir -p $DIR
echo "fixmean job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
run() { # $1=nome $2=step $3..=shards
  name=$1; step=$2; shift 2
  $PY scripts/exact_mean.py --dirs "$@" --out $DIR/$name --check /tmp/compose/${name}_merge 2>/dev/null | tail -2
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE "$EV" \
    --weights $DIR/$name --rows /tmp/mix/rows_holdout.npy --attn blas --batch 16 \
    > $DIR/ev_$name.txt 2>&1
  echo "  ev_$name ok"
}
run disj4 1952 /tmp/compose/disj4_s0/step_1952 /tmp/compose/disj4_s1/step_1952 /tmp/compose/disj4_s2/step_1952 /tmp/compose/disj4_s3/step_1952
run disj8 976 /tmp/compose/disj8_s{0,1,2,3,4,5,6,7}/step_976
run ord4 1952 /tmp/compose/ord4_s{0,1,2,3}/step_1952
run seed4 1952 /tmp/compose/seed4_s{0,1,2,3}/step_1952
run ann4 1952 /tmp/compose/ann4_s{0,1,2,3}/step_1952
run lr00002 1952 /tmp/compose/lr00002_s{0,1,2,3}/step_1952
run lr6e05 1952 /tmp/compose/lr6e05_s{0,1,2,3}/step_1952
echo "fim fixmean $(date '+%F %H:%M:%S')"
