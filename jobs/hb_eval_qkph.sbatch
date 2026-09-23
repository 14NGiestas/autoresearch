#!/bin/bash
#SBATCH -J evqkph
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /home/iangiestas/autoresearch-eval/evqkph_%j.log
# qkhop-ph no MESMO holdout do base f0 (2.1200): comparacao limpa.
set -u
EV=$HOME/autoresearch-eval
export ENERGY_LEDGER=$EV/energy/ledger.jsonl
echo "evqkph job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
python3 $EV/energy_run.py run --tag hb/eval_qkph --note "eval_bpb holdout 100 linhas" -- \
  env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \
  "$EV/portable/lib/ld-linux-x86-64.so.2" --library-path "$EV/portable/lib" "$EV/eval_bpb" --weights /tmp/w_qkph/step_7812 --rows $EV/rows_holdout.npy \
  --attn blas --batch 16 > $EV/ev_qkph_holdout.txt 2>&1
echo "fim evqkph $(date '+%F %H:%M:%S')"
