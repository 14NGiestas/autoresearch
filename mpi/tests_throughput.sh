#!/bin/bash
# mpi/tests_throughput.sh -- tokens/s do train_ddp com --attn blas (o default
# desde 2026-09-18), 4 threads por rank, lotes DISJUNTOS (--data rotate).
#
# Configuracoes (todas com o mesmo trabalho por passo; muda so' quem le' o que e
# a frequencia do coletivo):
#   A  -n 1  tau=1     B  -n 2  tau=1
#   C  -n 1  tau=100   D  -n 2  tau=100     (1 sync no fim, referencia de "sem sync")
# tau=1 e tau=100 sao os dois extremos: A/B isolam o custo do coletivo por passo,
# C/D dizem quanto sobra quando o coletivo nao aparece.
#
# Tambem imprime, por configuracao: tokens/s global (tokens/rank x ranks / wall),
# o tempo de CADA MPI_Allreduce (--sync_log 1) e o tamanho do buffer por sync.
#
# REGRAS DURAS: roda SO' dentro do Slurm (recusa sem SLURM_JOB_ID) e o wrapper
# jobs/fermi_mpithru.sbatch carrega --mem=4G + portao de concorrencia. mpirun
# fora do Slurm em paralelo com run longo = global_oom de 2026-09-18 07:18.
set -u
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "RECUSADO: mpirun so dentro do Slurm (use jobs/fermi_mpithru.sbatch, --mem=4G para -n 2)."
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
NSTEPS=${NSTEPS:-20}
TL="--weights $W --rows $ROWS --nsteps $NSTEPS --lr 1e-3 --ntrain 64 \
  --save_every 999 --data rotate --attn blas --ckpt-format npy --sync_log 1"
P="OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT OMP_DYNAMIC=FALSE"

echo "### host $(hostname)  $(date '+%F %H:%M:%S')  job ${SLURM_JOB_ID:-?}  mem_cap ${SLURM_MEM_PER_NODE:-?} MB"
echo "### bin $BIN  sha $(sha256sum "$BIN" | cut -c1-16)"
echo "### attn blas, threads/rank=$NT, passos=$NSTEPS, dados=rotate (lotes disjuntos)"

for tag in A B C D; do
  case $tag in
    A) np=1; tau=1 ;;
    B) np=2; tau=1 ;;
    C) np=1; tau=100 ;;
    D) np=2; tau=100 ;;
  esac
  rm -rf $S/thru$tag
  env $P mpirun -n $np "$BIN" $TL --out $S/thru$tag --sync grad --sync_every $tau \
      > $S/thru$tag.log 2>&1
  rc=$?
  wall=$(grep -o 'wall_s *[0-9.]*' $S/thru$tag.log | tail -1 | awk '{print $2}')
  tok=$(grep -o 'total *[0-9]*' $S/thru$tag.log | tail -1 | awk '{print $2}')
  ar=$(grep -o 'allreduce_s *[0-9.]*' $S/thru$tag.log | tail -1 | awk '{print $2}')
  tps=$(awk -v t="${tok:-0}" -v w="${wall:-0}" 'BEGIN{ if (w>0) printf "%.1f", t/w; else print "?" }')
  echo "--- $tag: -n $np, tau=$tau (exit $rc) ---"
  echo "    tokens total=$tok  wall=${wall}s  tokens/s=$tps  allreduce_total=${ar}s"
  grep -o 'rank . step . row .*fp [0-9A-F]*' $S/thru$tag.log | tail -2 | sed 's/^/    /'
  # estatistica dos MPI_Allreduce por chamada (--sync_log 1)
  if grep -q 'sync step' $S/thru$tag.log; then
    echo -n "    allreduce por chamada (s), ordenado: "
    grep 'sync step' $S/thru$tag.log | grep -o 'allreduce_s *[0-9.]*' | awk '{print $2}' | sort -n | paste -sd' ' -
    echo -n "    min/mediana/max (ms): "
    grep 'sync step' $S/thru$tag.log | grep -o 'allreduce_s *[0-9.]*' | awk '{print $2}' | sort -n | awk '
      {v[NR]=$1} END{ if (NR==0) {print "?"; exit}
        med=(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2;
        printf "%.1f / %.1f / %.1f  (n=%d chamadas)\n", 1000*v[1], 1000*med, 1000*v[NR], NR }'
  fi
  flush=1
done

# Tamanho do buffer de um sync: 8 arrays de GR (d216: 9.510.912 floats fp32).
echo
echo "### buffer por sync (dimensoes deste binario)"
python3 - <<'PY' 2>/dev/null || echo "  (python indisponivel; numeros da d216: 9.510.912 floats = 38.043.648 B)"
wte=8192*216; q=12*216*216; kv=12*(2*36)*216; fc=12*(4*216)*216
groups=[("wte",wte),("lm",wte),("q",q),("k",kv),("v",kv),("p",q),("fc",fc),("p2",fc)]
tot=sum(v for _,v in groups)
for n,v in groups:
    print(f"  GR[{n}] = {v:>10,} floats = {4*v/1e6:7.2f} MB".replace(",","."))
print(f"  TOTAL por sync = {tot:,} floats = {4*tot/1e6:.2f} MB = {4*tot/2**20:.2f} MiB em 8 coletivos".replace(",","."))
print(f"  (mesmo tamanho no --sync delta, sobre o delta dos pesos)")
PY
echo "### fim $(date '+%F %H:%M:%S')"
