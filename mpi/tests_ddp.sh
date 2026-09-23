#!/bin/bash
# mpi/tests_ddp.sh -- conjunto de evidencias do trainer MPI (mpi/app/train_ddp.f90).
#
# Roda SEMPRE dentro de um shell que tenha mpirun/mpif90 no PATH, por exemplo:
#   nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich -c bash mpi/tests_ddp.sh
# (jobs/fermi_mpiddp.sbatch faz exatamente isso dentro do SLURM.)
#
# Secoes:
#   1. refactor puro: codigo PRE-refactor (train_run na build arch_d216, release)
#      x train_ddp -n 1, 1 passo, mesmos dados -- os .npy tem de ser byte-identicos
#   2. INVARIANTE: -n 2 (sync grad, tau=1, chunk = pool inteiro, mesmos dados nos
#      dois ranks) x -n 1 -- byte-identico, passo a passo
#   3. todos os ranks bit-identicos entre si ao longo dos passos
#   4. --data rotate: lotes disjuntos, ranks ainda bit-identicos entre si
#   5. --sync delta: media de pesos no fim do bloco
#   6. mesma invariante com `--attn naive` (o kernel lento, agora OPCIONAL:
#      desde 2026-09-18 o default do train_ddp e' --attn blas, entao as secoes
#      2-5 exercitam o blas e esta garante que o naive continua coberto)
#
# O IMPOSTO de tau=1 em wall-clock fica em mpi/tests_tax.sh (mesmo computo,
# frequencia de sync diferente), para poder ser re-medido sozinho.
#
# Nada aqui toca src/build/arch_* (so LE o binario pre-refactor): o build MPI
# vive em mpi/build/ e o scratch em mpi/scratch_*/.
set -u

# REGRA DURA (incidente de 2026-09-18 07:18): mpirun/mpiexec NUNCA fora de um job
# do Slurm. Dois ranks do train_ddp (760-845 MB cada) em foreground entraram em
# paralelo com o job 103 (scal65) numa fermi de 15 GB e o global_oom do kernel
# matou o train_run (RSS 3.5 GB) -- 2 checkpoints de uma curva de 7h40 perdidos.
# Este portao torna a regra estrutural: sem SLURM_JOB_ID, o script se RECUSA a
# rodar; dentro do job use jobs/fermi_mpiddp.sbatch (--mem=4G para -n 2, 6G para
# -n 4, 10G para -n 8) e nunca em paralelo com um run longo.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "RECUSADO: mpirun so dentro do Slurm (use jobs/fermi_mpiddp.sbatch, --mem=4G para -n 2)."
  echo "Motivo: o mpirun em foreground de 2026-09-18 07:18 matou o job 103 num global_oom."
  exit 2
fi
cd "$(dirname "$0")/.."          # raiz do repo

# Falha RUIDOSA: nenhuma secao pode seguir depois de um run que nao terminou 0
# (foi assim que o job 114 saiu mudo com 4G de teto: a secao 1 morreu e as
# seguintes compararam diretorios que nao existiam). O log do run vai para a tela.
check_rc() { # check_rc <rc> <tag> <logfile>
  [ "$1" = 0 ] && return 0
  echo "  *** $2 FALHOU (exit $1) -- ultimas linhas de $3: ***"
  tail -8 "$3" | sed 's/^/    /'
  echo "ABORTA: evidencia invalida a partir daqui."
  exit 4
}

S=mpi/scratch_20260917
mkdir -p "$S"
BIN=$(ls -t mpi/build/*/app/train_ddp | head -1)
OLD=$(ls -t src/build/arch_d216_h6_kv2_l12_v8192_c1024/*/app/train_run | head -1)
W=/tmp/w_p2soup
ROWS=/tmp/mix/rows_dev.npy
BYTES=$HOME/.cache/autoresearch/tok_tables/token_bytes.txt

echo "### host $(hostname)  $(date '+%F %H:%M:%S')"
echo "### bin MPI  $BIN"
echo "### sha256 bin $(sha256sum "$BIN" | cut -c1-16)"
echo "### bin PRE  $OLD   ($(date -r "$OLD" '+%F %H:%M'))"
echo "### sha256 train_ddp.f90  $(sha256sum mpi/app/train_ddp.f90 | cut -c1-16)"
echo "### sha256 train.f90      $(sha256sum src/lib/fortran_train.f90 | cut -c1-16)"

tree_hash() { (cd "$1" && ls *.npy 2>/dev/null | sort | xargs cat | sha256sum | cut -c1-16); }
# cmpdir A B: compara todo *.npy de A com o de mesmo nome em B
cmpdir() {
  local a=$1 b=$2 bad=0 n=0 bn f
  for f in "$a"/*.npy; do
    bn=$(basename "$f")
    if [ ! -f "$b/$bn" ]; then echo "  FALTA  $b/$bn"; bad=$((bad + 1)); continue; fi
    if cmp -s "$f" "$b/$bn"; then n=$((n + 1)); else echo "  DIFERE $bn"; bad=$((bad + 1)); fi
  done
  echo "  -> $a x $b: $n arquivos byte-identicos, $bad diferentes"
  return $bad
}

# ---------------------------------------------------------------------------
echo
echo "=== 1. REFACTOR PURO: train_run (pre-refactor) x train_ddp -n 1, 1 passo ==="
rm -rf $S/ref_old $S/ref_new
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE "$OLD" \
  --weights $W --rows $ROWS --out $S/ref_old --nsteps 1 --lr 1e-3 --ntrain 8 \
  --start_row 0 --save_every 1 --val_every 999999 --nval 0 --trn_probe 1 \
  --attn naive --ckpt-format npy --bytes $BYTES > $S/ref_old.log 2>&1
check_rc $? ref_old $S/ref_old.log
echo "  train_run exit=$? ($(grep -c . $S/ref_old.log) linhas de log)"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE mpirun -n 1 "$BIN" \
  --weights $W --rows $ROWS --out $S/ref_new --nsteps 1 --lr 1e-3 --ntrain 8 \
  --chunk 8 --sync grad --sync_every 1 --data slice --save_every 1 \
  --attn naive --ckpt-format npy > $S/ref_new.log 2>&1
check_rc $? ref_new $S/ref_new.log
# (--attn naive EXPLICITO: o binario PRE-refactor usa o kernel naive por default;
#  desde 2026-09-18 o default do train_ddp e' blas, entao a comparacao bit a bit
#  da secao 1 tem de fixar o mesmo kernel dos dois lados.)
echo "  train_ddp exit=$? ($(grep -c . $S/ref_new.log) linhas de log)"
cmpdir $S/ref_old/step_1 $S/ref_new/step_1

# ---------------------------------------------------------------------------
# INVARIANTE: com --sync grad --sync_every 1 e os DOIS ranks lendo os MESMOS
# dados (--data slice, --chunk = pool = 8), cada rank faz a media de gradientes
# que e' o gradiente do MESMO lote => a sequencia de operacoes de um run de 1
# rank. Nada de aproximacao: tem de bater bit a bit.
COMMON="--weights $W --rows $ROWS --nsteps 3 --lr 1e-3 --ntrain 8 --chunk 8 \
  --save_every 3 --data slice --ckpt-format npy --nval 2 --val_every 3 --bytes $BYTES"
echo
echo "=== 2/3. INVARIANTE -n 2 x -n 1 (sync grad, tau=1, chunk = pool inteiro) ==="
rm -rf $S/inv2 $S/inv1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE mpirun -n 2 "$BIN" \
  $COMMON --out $S/inv2 --sync grad --sync_every 1 --save_all 1 \
  > $S/inv2.log 2>&1
check_rc $? inv2 $S/inv2.log
echo "  -n 2 exit=$?"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE mpirun -n 1 "$BIN" \
  $COMMON --out $S/inv1 --sync grad --sync_every 1 > $S/inv1.log 2>&1
check_rc $? inv1 $S/inv1.log
echo "  -n 1 exit=$?"
echo "  fingerprints por rank/passo (-n 2):"
grep '^rank .* step ' $S/inv2.log | sed 's/^/    /'
echo "  fingerprints por passo (-n 1):"
grep '^rank .* step ' $S/inv1.log | sed 's/^/    /'
echo "  tree_hash -n 2 rank_0 = $(tree_hash $S/inv2/rank_0/step_3)"
echo "  tree_hash -n 2 rank_1 = $(tree_hash $S/inv2/rank_1/step_3)"
echo "  tree_hash -n 1        = $(tree_hash $S/inv1/step_3)"
cmpdir $S/inv1/step_3 $S/inv2/rank_0/step_3
cmpdir $S/inv2/rank_0/step_3 $S/inv2/rank_1/step_3
grep 'final fp\|bit-identical\|tokens\|s/step\|allreduce\|sync tax' $S/inv2.log | sed 's/^/    /'

# ---------------------------------------------------------------------------
echo
echo "=== 4. --data rotate (lotes disjuntos por passo), -n 2, sync grad tau=1 ==="
rm -rf $S/rot2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE mpirun -n 2 "$BIN" \
  --weights $W --rows $ROWS --nsteps 3 --lr 1e-3 --ntrain 8 --save_every 3 \
  --data rotate --sync grad --sync_every 1 --ckpt-format npy --save_all 1 \
  --out $S/rot2 > $S/rot2.log 2>&1
check_rc $? rot2 $S/rot2.log
echo "  exit=$?"
grep '^rank .* step ' $S/rot2.log | sed 's/^/    /'
grep 'final fp\|bit-identical' $S/rot2.log | sed 's/^/    /'
echo "  (esperado: rank0 e rank1 com o MESMO fp em cada passo -- mesmo update --"
echo "   mas os dados sao lidos de linhas diferentes dos passos 1..3)"

# ---------------------------------------------------------------------------
echo
echo "=== 5. --sync delta (local SGD, media de pesos no fim do bloco), tau=3 ==="
rm -rf $S/delta2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE mpirun -n 2 "$BIN" \
  --weights $W --rows $ROWS --nsteps 3 --lr 1e-3 --ntrain 8 --save_every 3 \
  --data rotate --sync delta --sync_every 3 --ckpt-format npy --save_all 1 \
  --out $S/delta2 > $S/delta2.log 2>&1
check_rc $? delta2 $S/delta2.log
echo "  exit=$?"
grep '^rank .* step ' $S/delta2.log | sed 's/^/    /'
grep 'final fp\|bit-identical' $S/delta2.log | sed 's/^/    /'
cmpdir $S/delta2/rank_0/step_3 $S/delta2/rank_1/step_3

# ---------------------------------------------------------------------------
echo
echo "=== 6. invariante com --attn naive (kernel lento, agora opcional) ==="
rm -rf $S/naive2 $S/naive1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE mpirun -n 2 "$BIN" \
  $COMMON --out $S/naive2 --sync grad --sync_every 1 --attn naive > $S/naive2.log 2>&1
check_rc $? naive2 $S/naive2.log
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE mpirun -n 1 "$BIN" \
  $COMMON --out $S/naive1 --sync grad --sync_every 1 --attn naive > $S/naive1.log 2>&1
check_rc $? naive1 $S/naive1.log
echo "  tree_hash -n 2 = $(tree_hash $S/naive2/step_3)   -n 1 = $(tree_hash $S/naive1/step_3)"
cmpdir $S/naive1/step_3 $S/naive2/step_3


echo
echo "### fim $(date '+%F %H:%M:%S')"
