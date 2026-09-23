#!/bin/bash
# mpi/tests_quant.sh -- DESVIO do coletivo de 16 bits (--sync-fp16 bf16|fix).
#
# DOIS EIXOS, explicitos:
#   EIXO 1 (invariante DENTRO do modo): rank_0 x rank_1 do MESMO modo (ref, fix,
#     bf16) tem de ser byte-identico -- a quantizacao e' uma mudanca de
#     trajetoria, nao uma divergencia entre ranks (o buffer quantizado sai do
#     coletivo, nao do arredondamento local).
#   EIXO 2 (desvio ENTRE modos, MESMO conjunto de dados): fix e bf16 contra ref
#     (fp32 bit-exato) -- desvio numerico dos pesos e desvio de qualidade (bpb).
#
# Os 3 runs tem config IDENTICA: -n 2, --sync grad --sync_every 1, --chunk = pool
# (os dois ranks leem os MESMOS dados), --attn blas, 4 threads/rank, NSTEPS passos
# com checkpoints em NSTEPS/2 e NSTEPS. A unica diferenca entre eles e' o modo do
# coletivo -- logo o eixo 2 mede so' o efeito da quantizacao.
#
# LICOES DOS JOBS 114/116 (que falharam):
#   - python: SEMPRE /home/pauli/autoresearch/.venv-numpy/bin/python3 (o python do
#     nix shell nao tem numpy; o job 116 morreu no primeiro import e o script
#     seguiu adiante comparando diretorios que nao existiam);
#   - falha tem de ser RUIDOSA: rc de cada run checado, log do run impresso e
#     abort imediato se o checkpoint esperado nao existe (nada de "0 identicos,
#     1 diferentes" de glob nao expandido).
#
# REGRAS: so' dentro do Slurm (jobs/fermi_mpiquant.sbatch, teto de 6G via
# mpi/guard.sh); nunca em paralelo com run longo.
set -u
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "RECUSADO: mpirun so' dentro do Slurm (use jobs/fermi_mpiquant.sbatch)."
  exit 2
fi
cd "$(dirname "$0")/.."
S=mpi/scratch_20260917
mkdir -p "$S"
PY=/home/pauli/autoresearch/.venv-numpy/bin/python3
BIN=$(ls -t mpi/build/*/app/train_ddp | head -1)
W=/tmp/w_p2soup
BYTES=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt
NT=${NT:-4}
NSTEPS=${NSTEPS:-40}
SAVE=$((NSTEPS / 2))
NVAL=${NVAL:-300}
NTRAIN=64
ROWS=$S/rows_quant.npy
MODES="ref fix bf16"

[ -x "$PY" ] || { echo "ABORTA: $PY nao existe/nao e' executavel"; exit 3; }

# Holdout: 64 linhas de treino + 300 de holdout no MESMO arquivo (o app le treino
# em [start_row, start_row+ntrain) e val em [start_row+ntrain, +nval)).
if [ ! -f "$ROWS" ]; then
  "$PY" - "$ROWS" <<'EOF' || { echo "ABORTA: falhou ao montar $ROWS"; exit 3; }
import sys
import numpy as np
tr = np.load('/tmp/mix/rows_dev.npy')          # 100 x 1025
va = np.load('/tmp/prose65/rows_val.npy')      # 300 x 1025
out = np.concatenate([tr[:64], va[:300]]).astype(np.int32)
np.save(sys.argv[1], out)
print(f"  rows_quant.npy: {out.shape} = 64 treino + 300 holdout")
EOF
fi
[ -s "$ROWS" ] || { echo "ABORTA: $ROWS vazio"; exit 3; }

COMMON="--weights $W --rows $ROWS --nsteps $NSTEPS --lr 1e-3 --ntrain $NTRAIN \
  --start_row 0 --save_every $SAVE --data slice --chunk $NTRAIN --attn blas \
  --ckpt-format npy --nval $NVAL --val_every $NSTEPS --bytes $BYTES --sync_log 1"
P="OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT OMP_DYNAMIC=FALSE"

echo "### host $(hostname)  $(date '+%F %H:%M:%S')  job ${SLURM_JOB_ID:-?}"
echo "### bin $BIN  sha $(sha256sum "$BIN" | cut -c1-16)"
echo "### python $PY  ($("$PY" -c 'import numpy; print("numpy", numpy.__version__)'))"
echo "### -n 2, sync grad tau=1, mesmos dados nos 2 ranks, blas, $NT threads/rank,"
echo "### $NSTEPS passos (checkpoints em $SAVE e $NSTEPS), holdout $NVAL linhas"
echo "### EIXO 1 = rank_0 x rank_1 DENTRO do modo | EIXO 2 = fix/bf16 vs ref"

case_of() { case $1 in ref) echo none;; fix) echo fix;; bf16) echo bf16;; esac; }

for tag in $MODES; do
  rm -rf $S/q_$tag
  env $P mpirun -n 2 "$BIN" $COMMON --out $S/q_$tag --save_all 1 \
      --sync-fp16 "$(case_of $tag)" > $S/q_$tag.log 2>&1
  rc=$?
  echo "--- $tag (--sync-fp16 $(case_of $tag)) exit=$rc ---"
  grep -E 'sync-fp16|payload 16b|^val @' $S/q_$tag.log | sed 's/^/    /'
  grep -E 'bit-identical' $S/q_$tag.log | sed 's/^/    /'
  grep -o 'wall_s *[0-9.]*\|allreduce_s *[0-9.]*' $S/q_$tag.log | tail -2 | paste -sd' ' - | sed 's/^/    /'
  echo -n "    allreduce por chamada (ms, ordenado): "
  grep 'sync step' $S/q_$tag.log | grep -o 'allreduce_s *[0-9.]*' | awk '{printf "%.2f\n", 1000*$2}' | sort -n | paste -sd' ' -
  if [ $rc -ne 0 ]; then
    echo "    *** FALHOU (exit $rc) -- ultimas linhas do log: ***"
    tail -12 $S/q_$tag.log | sed 's/^/      /'
    echo "ABORTA: sem run valido de $tag nao ha' o que comparar."
    exit 4
  fi
  for step in $SAVE $NSTEPS; do
    for r in 0 1; do
      d=$S/q_$tag/rank_$r/step_$step
      nf=$(ls $d/*.npy 2>/dev/null | wc -l)
      if [ "$nf" -ne 90 ]; then
        echo "ABORTA: $d tem $nf .npy (esperado 90) -- checkpoint incompleto."
        exit 4
      fi
    done
  done
done

echo
echo "=== EIXO 1: invariante DENTRO do modo (rank_0 x rank_1, cmp byte a byte) ==="
for tag in $MODES; do
  for step in $SAVE $NSTEPS; do
    n=0; bad=0
    for f in $S/q_$tag/rank_0/step_$step/*.npy; do
      b=$(basename "$f")
      if cmp -s "$f" "$S/q_$tag/rank_1/step_$step/$b"; then n=$((n+1)); else echo "  DIFERE $tag/$b"; bad=$((bad+1)); fi
    done
    echo "  $tag passo $step: rank_0 x rank_1 -> $n identicos, $bad diferentes"
  done
done

echo
echo "=== EIXO 2a: desvio NUMERICO dos pesos (fix/bf16 contra ref, mesmo modo de dados) ==="
"$PY" - "$S" $SAVE $NSTEPS <<'EOF' || { echo "ABORTA: analise numerica falhou"; exit 5; }
import sys, os, glob
import numpy as np
S = sys.argv[1]; steps = sys.argv[2:]
def load(d):
    out = {}
    for p in glob.glob(d + "/*.npy"):
        b = os.path.basename(p)
        if b.startswith(("adam_", "muon_")):
            continue
        out[b] = np.load(p).astype(np.float64)
    return out
for step in steps:
    R = load(f"{S}/q_ref/rank_0/step_{step}")
    print(f"  passo {step} ({len(R)} tensores de peso):")
    for tag in ("fix", "bf16"):
        Q = load(f"{S}/q_{tag}/rank_0/step_{step}")
        rel, absmax, num, den = [], 0.0, 0.0, 0.0
        worst = (0.0, "")
        for k in sorted(R):
            if k not in Q:
                raise SystemExit(f"falta {k} em q_{tag}/step_{step}")
            w, q = R[k], Q[k]
            dw = float(np.abs(q - w).max())
            absmax = max(absmax, dw)
            wm = float(np.abs(w).max())
            if wm > 0:
                r = dw / wm
                rel.append(r)
                if r > worst[0]:
                    worst = (r, k)
            num += float(((q - w)**2).sum()); den += float((w**2).sum())
        rel = np.asarray(rel)
        print(f"    {tag:>4} vs ref: max|dw|/max|w| por tensor = mediana {np.median(rel):.3e}  "
              f"pior {worst[0]:.3e} ({worst[1]})  ||dw||/||w|| = {np.sqrt(num/den):.3e}  "
              f"max|dw| = {absmax:.3e}")
EOF

echo
echo "=== EIXO 2b: desvio de QUALIDADE (bpb no MESMO holdout de $NVAL linhas) ==="
for tag in $MODES; do
  grep -E '^val @' $S/q_$tag.log | tail -1 | sed "s/^/  $tag: /"
done
echo "  (mesmas $NVAL linhas, mesmos pesos iniciais, mesma config de treino:"
echo "   a diferenca entre ref e fix/bf16 e' so' a quantizacao do coletivo)"
echo "### fim $(date '+%F %H:%M:%S')"
