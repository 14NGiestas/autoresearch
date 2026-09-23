#!/bin/bash
#SBATCH -J archhead
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o /home/pauli/autoresearch/logs/archhead_%j.log
# VARIAVEL DE ARQUITETURA: head_dim (hyp_ad78cc).
#
# Motivacao medida no job 149/antes disso: a atencao domina o passo e roda a
# 9.8 GFLOP/s com head_dim=16 contra 74.3 GFLOP/s com head_dim=128 no mesmo T --
# os FLOPs sao os MESMOS (2*T^2*(H*D), com H*D=96 fixo), so' a FORMA muda. Se o
# passo cair sem perder bpb, o gargalo e' arquitetural e o caminho esta' aberto.
#
# Variantes (d_model=96, 12 camadas, 8192 vocab): (heads,kv) = (6,2) [atual],
# (3,1), (2,1) -> head_dim 16, 32, 48. kv escolhido para manter d_kv ~= 32
# (params praticamente iguais; o script imprime a diferenca).
#
# Controles: MESMA semente no make_init (init aleatorio identico em espirito),
# MESMOS tokens (1M = 1000 passos a B=1,T=1024), OMP_NUM_THREADS=8 fixo (medir
# com 16 threads da' numeros 2.3x piores -- medido), e a identidade da arch
# impressa por bin/build_arch em cada variante.
set -u
cd /home/pauli/autoresearch
export OMP_NUM_THREADS=8
PY=.venv-numpy/bin/python3
BY="$HOME/.cache/autoresearch/tok_tables/token_bytes.txt"
ROWS=/tmp/mix/rows_f0.npy
HOLD=/tmp/mix/rows_holdout.npy
NST=1000
echo "archhead job=${SLURM_JOB_ID} $(date '+%F %H:%M:%S') load=$(cut -d' ' -f1 /proc/loadavg)"

printf "%-14s %-9s %-10s %-10s %-10s\n" "variante" "head_dim" "identidade" "s/passo" "bpb(holdout)"
for V in "6 2" "3 1" "2 1"; do
  set -- $V; H=$1; KV=$2
  HD=$((96/H))
  OUT=/tmp/arch/h${H}_kv${KV}
  rm -rf "$OUT"
  $PY scripts/make_init.py --out "$OUT" --d 96 --heads "$H" --kv "$KV" --layers 12 \
      --vocab 8192 --ctx 1024 --seed 7 >/dev/null || { echo "  H=$H: init falhou"; continue; }
  BD=/tmp/arch/build_h${H}
  rm -rf "$BD"
  bin/build_arch 96 "$H" "$KV" 12 8192 1024 "$BD" > "$BD.buildlog" 2>&1
  ID=$(grep -E '^  id [0-9a-f]{16}' "$BD.buildlog" | awk '{print $2}' | tail -1)
  if [ -z "${ID:-}" ]; then
    echo "  H=$H kv=$KV: BUILD FALHOU (ver $BD.buildlog)"; sed -n '$p' "$BD.buildlog" | sed 's/^/    /'; continue
  fi
  bin/job_env --build-dir "$BD" "--rows:src/app/train_run.f90" >/dev/null || exit 2
  . "$BD/job_env.sh"
  BIN="$TRAIN_RUN"
  EVB="$EVAL_BPB"
  [ -z "$BIN" ] && { echo "  H=$H: train_run ausente"; continue; }
  T0=$(date +%s.%N)
  $BIN --weights "$OUT" --rows $ROWS --out "$OUT/run" --nsteps $NST --lr 6e-4 \
      --ntrain 11797 --start_row 0 --nval 1 --val_every 9999999 --trn_probe 1 \
      --save_every $NST --attn blas --batch 1 --bytes "$BY" > "$OUT/train.log" 2>&1
  T1=$(date +%s.%N)
  SP=$(echo "($T1-$T0)/$NST" | bc -l | cut -c1-5)
  CK=$(ls -d "$OUT"/run/step_* 2>/dev/null | tail -1)
  B=$($EVB --weights "$CK" --rows $HOLD --batch 16 --attn blas 2>/dev/null | grep -oE "[0-9]\.[0-9]{4,}" | tail -1)
  printf "%-14s %-9s %-10s %-10s %-10s\n" "h${H}_kv${KV}" "$HD" "${ID:-?}" "$SP" "${B:-n/d}"
done

echo
echo "# leitura: s/passo deve CAIR com head_dim maior (a atencao e' a mesma conta em"
echo "#   forma melhor). bpb deve ficar ~igual (1M tokens, mesma semente de init); se"
echo "#   cair junto, melhor ainda. Se o s/passo nao cair, a hipotese esta' errada."
echo "fim archhead $(date '+%F %H:%M:%S')"
