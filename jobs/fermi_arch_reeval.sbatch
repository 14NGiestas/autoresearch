#!/bin/bash
#SBATCH -J archreeval
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/archreeval_%j.log
# Re-evaluate the arms of archqual with the canonical bpb rule.
#
# The bpb column of job 154 is wrong. The app eval_bpb prints the negative log
# likelihood of each token and no summary, and the job parsed the last number of
# the last row: it reported 9.49 for a model at 3.78 nats. The rule lives in
# bpb_of() of scripts/fedavg_rounds.py, and scripts/eval_bpb_dir.py uses it.
#
# This job runs after 154, because an eval uses the CPU and job 154 measures the
# seconds for each step.
set -u
cd /home/pauli/autoresearch
export OMP_NUM_THREADS=4
PY=.venv-numpy/bin/python3
echo "archreeval job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S')"
printf "%-16s %-10s %-10s\n" "arm" "head_dim" "bpb"
for RUN in /tmp/aq/h*_s*/run/step_*; do
  [ -d "$RUN" ] || continue
  TAG=$(echo "$RUN" | sed 's|/tmp/aq/||; s|/run/.*||')
  H=$(echo "$TAG" | sed 's|h\([0-9]*\)_.*|\1|')
  EVB=$(ls /tmp/aq/build_h${H}_kv*/*/app/eval_bpb 2>/dev/null | tail -1)
  [ -z "$EVB" ] && { printf "%-16s %-10s %-10s\n" "$TAG" "$((96/H))" "sem eval"; continue; }
  B=$($PY scripts/eval_bpb_dir.py --eval "$EVB" --ckpt "$RUN" 2>/dev/null | tail -1)
  printf "%-16s %-10s %-10s\n" "$TAG" "$((96/H))" "${B:-falhou}"
done
echo "fim archreeval $(date '+%F %H:%M:%S')"
