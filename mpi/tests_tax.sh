#!/bin/bash
# mpi/tests_tax.sh -- quanto custa tau=1, em wall-clock, no trainer MPI.
#
# Isola SINCRONIZACAO de COMPUTO: todos os runs fazem o MESMO numero de passos
# com os MESMOS dados (--data rotate: lotes disjuntos, um lote por rank por
# passo), o mesmo numero de threads por rank e o mesmo numero de ranks. O que
# muda entre A, B e D e' so' a FREQUENCIA do coletivo (tau).
#   A  -n 2  sync grad  tau=1  (6 syncs em 6 passos)
#   D  -n 2  sync delta tau=3  (2 syncs em 6 passos)
#   B  -n 2  sync delta tau=6  (1 sync em 6 passos)
#   C  -n 1  sync grad  tau=1  (referencia de 1 rank: sem coletivo)
# O proprio app cronometra o MPI_Allreduce (allreduce_s) separado do passo.
#
# ATENCAO a maquina: o login node E' o node de computo (fermi, 16 CPUs) e o job
# 103 do usuario ocupa 8 delas. Os segundos absolutos sao indicativos; a
# DIFERENCA entre A/B/D e o allreduce_s medido pelo app e' o que se le.
#
#   nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich -c bash mpi/tests_tax.sh
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
cd "$(dirname "$0")/.."

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
W=/tmp/w_p2soup
ROWS=/tmp/mix/rows_dev.npy
NT=${NT:-4}
NSTEPS=${NSTEPS:-6}
TL="--weights $W --rows $ROWS --nsteps $NSTEPS --lr 1e-3 --ntrain 8 --save_every 999 \
  --data rotate --ckpt-format npy"
P="OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT OMP_DYNAMIC=FALSE"

echo "### host $(hostname)  $(date '+%F %H:%M:%S')  uptime:$(uptime | sed 's/.*up //')"
echo "### bin $BIN  threads/rank $NT  steps $NSTEPS"
echo "### sha256 bin $(sha256sum "$BIN" | cut -c1-16)"
for tag in A D B C; do
  case $tag in
    A) args="--out $S/taxA --sync grad --sync_every 1"; np=2; msg="A: 2 ranks, sync grad tau=1" ;;
    D) args="--out $S/taxD --sync delta --sync_every 3"; np=2; msg="D: 2 ranks, sync delta tau=3" ;;
    B) args="--out $S/taxB --sync delta --sync_every $NSTEPS"; np=2; msg="B: 2 ranks, sync delta tau=$NSTEPS" ;;
    C) args="--out $S/taxC --sync grad --sync_every 1"; np=1; msg="C: 1 rank, sync grad tau=1" ;;
  esac
  rm -rf $S/tax$tag
  env $P mpirun -n $np "$BIN" $TL $args > $S/tax$tag.log 2>&1
  echo "--- $msg (exit $?) ---"
  grep 'wall_s\|s/step\|final fp' $S/tax$tag.log | sed 's/^/    /'
done

get() { grep -o "$2 *[0-9.]*" $S/tax$1.log | tail -1 | awk '{print $2}'; }
for tag in A D B C; do
  w=$(get $tag wall_s); c=$(get $tag allreduce_s)
  [ -n "$w" ] && echo "  $tag: wall=$w s  allreduce=$c s  allreduce/wall=$(awk -v a="$c" -v b="$w" 'BEGIN{printf "%.3f%%", 100*a/b}')"
done
wa=$(get A wall_s); wb=$(get B wall_s); wd=$(get D wall_s); wc=$(get C wall_s)
ca=$(get A allreduce_s); cb=$(get B allreduce_s); cd=$(get D allreduce_s)
echo "  (A-B) via wall: $(awk -v a="$wa" -v b="$wb" -v n="$NSTEPS" 'BEGIN{printf "%.3f", (a-b)/n}') s/passo"
echo "  (A-D)/4 passos com 4 syncs a menos: $(awk -v a="$ca" -v d="$cd" 'BEGIN{printf "%.3f", (a-d)/4}') s/sync (medido no app)"
echo "  A allreduce/$NSTEPS passos: $(awk -v a="$ca" -v n="$NSTEPS" 'BEGIN{printf "%.3f", a/n}') s/sync   B (1 sync): $cb s"
echo "  computo puro (C, 1 rank, sem coletivo): $wc s total"

# ---------------------------------------------------------------------------
# E: 1 thread por rank (sem contencao de CPU com o job 103) e --sync_log 1:
# um numero por COLETIVO. Separa o custo real do coletivo (o minimo observado)
# da contencao de CPU (o processo desescalonado paga segundos de wall).
echo
echo "--- E: 2 ranks x 1 thread, tau=1, 3 passos, --sync_log 1 (por coletivo) ---"
rm -rf $S/taxE
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE mpirun -n 2 "$BIN" \
  --weights $W --rows $ROWS --nsteps 3 --lr 1e-3 --ntrain 8 --save_every 999 \
  --data rotate --ckpt-format npy --out $S/taxE --sync grad --sync_every 1 \
  --sync_log 1 > $S/taxE.log 2>&1
check_rc $? taxE $S/taxE.log
grep 'sync step\|final fp\|wall_s\|s/step' $S/taxE.log | sed 's/^/    /'
echo "  tempos de UM coletivo (38 MB, 2 ranks/1 no), ordenados:"
grep -o 'allreduce_s *[0-9.]*' $S/taxE.log | awk '{print $2}' | sort -n | sed 's/^/    /' | paste -sd' ' -
echo "### fim $(date '+%F %H:%M:%S')"
