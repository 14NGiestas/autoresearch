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
# Duas licoes que custaram o job 155 (2 de 6 bracos "eval failed"):

#  1. NAO procurar binario por glob com `tail -1`. O glob build_h6_kv* casava
#     com kv1 E kv2, e o ultimo alfabetico e' o kv2 -- entao os bracos kv=1
#     foram avaliados com binario de kv=2 e o check de arch do loader abortou
#     (com razao: a infraestrutura de identidade funcionou). O jeito certo e'
#     LER o kv da propria checkpoint em arch.txt e pedir exatamente aquele.
#  2. NAO engolir stderr com 2>/dev/null. A causa era uma linha de arch
#     mismatch; escondida, virou "eval failed" e custou uma investigacao.

N_OK=0; N_TOT=0
for RUN in /tmp/aq/h*_s*/run/step_*; do
  [ -d "$RUN" ] || continue
  N_TOT=$((N_TOT + 1))
  TAG=$(echo "$RUN" | sed 's|/tmp/aq/||; s|/run/.*||')
  # O kv e o n_head vem do PROPRIO checkpoint, e o caminho do binario vem do
  # bin/build_arch (que publica "[build_arch] app eval_bpb <path>"). Nada de
  # glob com tail -1: o job 155 avaliou dois bracos kv=1 com binario kv=2.
  # Bonus: build_arch CONSTROI o binario que falta, que era o buraco raiz.
  KVF=$(sed -n 's/.*n_kv *= *\([0-9]*\).*/\1/p' "$RUN/arch.txt" 2>/dev/null | head -1)
  HF=$(sed -n 's/.*n_head *= *\([0-9]*\).*/\1/p' "$RUN/arch.txt" 2>/dev/null | head -1)
  [ -z "$KVF" ] && { printf "%-16s %-10s %s\n" "$TAG" "?" "sem arch.txt"; continue; }
  BD=/tmp/aq/build_h${HF}_kv${KVF}
  EVB=$(bin/build_arch 96 "$HF" "$KVF" 12 8192 1024 "$BD" 2>&1 | sed -n 's/^\[build_arch\] app eval_bpb //p')
  [ -z "$EVB" ] && { printf "%-16s %-10s %s\n" "$TAG" "$((96/HF))" "sem binario eval"; continue; }
  OUT=$($PY scripts/eval_bpb_dir.py --eval "$EVB" --ckpt "$RUN" 2>&1 | tail -1)
  # um bpb sempre comeca com digito; qualquer outra coisa e' erro, e o erro
  # aparece inteiro em vez de virar "falhou".
  case "$OUT" in
    [0-9]*) printf "%-16s %-10s %-10s\n" "$TAG" "$((96/HF))" "$OUT"
            N_OK=$((N_OK + 1)) ;;
    *)      printf "%-16s %-10s %s\n" "$TAG" "$((96/HF))" "FALHOU: $OUT" ;;
  esac
done
# cobertura: uma tabela parcial nao pode passar por completa
echo "cobertura: $N_OK de $N_TOT bracos avaliados"
[ "$N_OK" -lt "$N_TOT" ] && { echo "INCOMPLETO: faltam $((N_TOT - N_OK))"; exit 4; }
echo "fim archreeval $(date '+%F %H:%M:%S')"
