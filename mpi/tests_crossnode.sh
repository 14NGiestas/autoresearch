#!/bin/bash
# mpi/tests_crossnode.sh -- cross-node REAL: fermi (rank 0) + halfbeast (rank 1).
#
# A FERMI ESTA' ATRAS DE NAT (192.168.180.177): o halfbeast nao alcanca esse IP.
# O hydra precisa de dois canais que atravessem isso, e os dois sao tunelados em
# SSH (ver mpi/crossnode_sshwrap.sh e mpi/crossnode_setup.sh):
#   1. CONTROLE: o hydra_pmi_proxy remoto conecta de volta no
#      --control-port do launcher -> reescrito para 127.0.0.1:<porta> + "-R"
#      no ssh (wrapper usado via -launcher-exec).
#   2. DADOS: rank 0 <-> rank 1 pelo ch3:tcp -> faixa de portas FIXA
#      (CN_PORT_LO..CN_PORT_HI) + MPIR_CVAR_CH3_INTERFACE_HOSTNAME=127.0.0.1
#      + tuneis -L/-R para a faixa.
#
# Sequencia: (0) staging; (1) hostname (so' canal de controle);
# (2) allreduce_probe de 1 MB (caminho de DADOS); (3) invariante com dados
# IGUAIS vs 1 rank local; (4) throughput 1 rank local x 2 ranks cross, tau=1/100.
#
# SEGURANCA (requisito, nao preferencia): TODO mpirun usa -envlist EXPLICITO.
# Ambiente herdado faz o hydra encaminhar o ambiente inteiro (token de API, chave
# de terminal) para o no remoto. Nada de -envall/-genvall.
#
# So' roda dentro do Slurm (jobs/fermi_crossnode.sbatch, teto 6G via mpi/guard.sh).
set -u
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "RECUSADO: so' dentro do Slurm (jobs/fermi_crossnode.sbatch)."; exit 2
fi
cd "$(dirname "$0")/.."
STAGE=${STAGE:-/tmp/ddp}
HOSTS=$STAGE/hosts
BIN=$STAGE/train_ddp                 # MESMO caminho nos dois nos
PROBE=$STAGE/allreduce_probe
WRAP=$STAGE/crossnode_sshwrap.sh
ROWS=$STAGE/rows.npy
INIT=$STAGE/init
BYTES=$STAGE/token_bytes.txt
OUT=$STAGE/out
NT=${NT:-4}
NSTEPS=${NSTEPS:-20}
MPC="PATH,LD_LIBRARY_PATH,HOME,TMPDIR,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS"
CVARS="MPIR_CVAR_CH3_INTERFACE_HOSTNAME,MPIR_CVAR_CH3_PORT_RANGE,MPIR_CVAR_CH3_NETWORK_IFACE,MPIR_CVAR_NETWORK_IFACE,MPIR_CVAR_NEMESIS_TCP_NETWORK_IFACE"
ENVF="$MPC,$CVARS"
PY=/home/pauli/autoresearch/.venv-numpy/bin/python3
SSH="ssh -o BatchMode=yes"
P="OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT OMP_DYNAMIC=FALSE"

check_rc() { [ "$1" = 0 ] && return 0
  echo "  *** $2 FALHOU (exit $1) -- ultimas linhas de $3:"; tail -12 "$3" | sed 's/^/    /'; return 1; }
loads() { echo "    load fermi: $(uptime | sed 's/.*load average: //') | halfbeast: $($SSH halfbeast 'uptime | sed "s/.*load average: //"')"; }

echo "### host $(hostname)  $(date '+%F %H:%M:%S')  job ${SLURM_JOB_ID}"
echo "### bin $BIN sha $(sha256sum "$BIN" | cut -c1-16)   probe sha $(sha256sum "$PROBE" | cut -c1-16)"
echo "### hosts: $(tr '\n' ' ' < $HOSTS)"
echo "### NAT: fermi=$(hostname -I | awk '{print $1}') (privado)  -> precisa de tunel ssh"
loads

# ---------------------------------------------------------------------------
echo
echo "=== (0) staging + tuneis ==="
bash mpi/crossnode_stage.sh 2>&1 | sed 's/^/  /' || { echo "ABORTA: staging falhou"; exit 3; }
source mpi/crossnode_setup.sh
crossnode_setup || { echo "ABORTA: tuneis"; exit 3; }

MP_BASE="mpirun -f $HOSTS -ppn 1 -n 2 -launcher ssh -envlist $ENVF"
MPL="mpirun -n 1 -envlist $ENVF"
echo "### COMANDO CROSS (receita de controle resolvida em (1) abaixo)"

# ---------------------------------------------------------------------------
echo
echo "=== (1) canal de CONTROLE: hostname nos 2 nos ==="
echo "  -- 1a SEM wrapper (deve reproduzir o erro; HYDRA_LAUNCHER_EXTRA_ARGS=-v mostra o comando) --"
HYDRA_LAUNCHER_EXTRA_ARGS='-v' timeout 90 mpirun -f $HOSTS -ppn 1 -n 2 -envlist $MPC hostname \
  > $STAGE/t0.log 2>&1
grep -iE "launch proxy|callback|Permission|refused|No route|timed out|control-port" $STAGE/t0.log | head -6 | sed 's/^/     /'
echo "  -- 1b receitas COM o wrapper de tunel --"
MPX=""
for rec in "-iface lo -launcher-exec $WRAP" \
           "-localhost 127.0.0.1 -launcher-exec $WRAP" \
           "-iface lo -launcher-exec $WRAP -hydra-launcher-extra-args none"; do
  echo "     receita: $rec"
  if env $P $MP_BASE $rec hostname > $STAGE/t1.log 2>&1; then
    echo "     OK -> hosts: $(sort $STAGE/t1.log | tr '\n' ' ')"
    MPX="$MP_BASE $rec"
    break
  fi
  grep -iE "launch proxy|callback|error|refused|invalid|iface" $STAGE/t1.log | head -3 | sed 's/^/       /'
done
if [ -z "$MPX" ]; then echo "     *** nenhuma receita de controle funcionou ***"; fi

# ---------------------------------------------------------------------------
echo
echo "=== (2) caminho de DADOS: allreduce_probe (20x 1 MB) ==="
env $P $MPX $PROBE > $STAGE/t2.log 2>&1; rc=$?
if check_rc $rc "probe-cross" $STAGE/t2.log; then
  grep -E '^rank ' $STAGE/t2.log | sed 's/^/    /'
  EXIT_OK=1
else
  echo "    probe cross-node nao passou -- tentando variantes de env do netmod"
  for extra in \
    "-env MPIR_CVAR_CH3_INTERFACE_HOSTNAME 127.0.0.1 -env MPIR_CVAR_CH3_PORT_RANGE $CN_PORT_LO:$CN_PORT_HI" \
    "-env MPIR_CVAR_CH3_NETWORK_IFACE lo -env MPIR_CVAR_NETWORK_IFACE lo -env MPIR_CVAR_CH3_INTERFACE_HOSTNAME 127.0.0.1"; do
    echo "    --- variante: $extra"
    timeout 120 env $P mpirun -f $HOSTS -ppn 1 -n 2 -launcher ssh -launcher-exec $WRAP \
      -iface lo -envlist $MPC $extra $PROBE 2>&1 | grep -E '^rank |error|Error|failed' | head -6 | sed 's/^/      /'
  done
fi

# ---------------------------------------------------------------------------
echo
echo "=== (3) INVARIANTE cross-node: 2 ranks (1 por no), dados IGUAIS, tau=1 ==="
COMMON="--weights $INIT --rows $ROWS --attn blas --lr 1e-3 --ntrain 64 \
  --start_row 0 --data slice --chunk 64 --ckpt-format npy --bytes $BYTES --sync_log 1"
rm -rf $OUT/inv_loc $OUT/inv_x
env $P $MPL $BIN $COMMON --out $OUT/inv_loc --nsteps 3 --save_every 3 \
    --sync grad --sync_every 1 > $STAGE/b1.log 2>&1; check_rc $? b1_local $STAGE/b1.log
env $P $MPX $BIN $COMMON --out $OUT/inv_x --nsteps 3 --save_every 3 \
    --sync grad --sync_every 1 > $STAGE/b2.log 2>&1; check_rc $? b2_cross $STAGE/b2.log
echo "  fingerprints (1 rank local x 2 ranks cross-node):"
grep -h '^rank .* step ' $STAGE/b1.log $STAGE/b2.log | sed 's/^/    /'
grep -h 'bit-identical' $STAGE/b2.log | sed 's/^/    /'
echo "  cmp do checkpoint (rank 0 cross x 1 rank local):"
if [ -d $OUT/inv_loc/step_3 ] && [ -d $OUT/inv_x/step_3 ]; then
  n=0; bad=0
  for f in $OUT/inv_loc/step_3/*.npy; do
    b=$(basename "$f")
    if cmp -s "$f" "$OUT/inv_x/step_3/$b"; then n=$((n+1)); else echo "  DIFERE $b"; bad=$((bad+1)); fi
  done
  echo "  -> $n identicos, $bad diferentes"
else
  echo "  (sem checkpoint dos dois lados: cross-node nao completou)"
fi
grep -h 'allreduce_s' $STAGE/b2.log | tail -1 | sed 's/^/    cross allreduce: /'

# ---------------------------------------------------------------------------
echo
echo "=== (4) throughput: 1 rank local x 2 ranks cross, tau=1 e tau=100 ==="
TL="--nsteps $NSTEPS --save_every 999 --data rotate"
run_t() { # run_t <tag> <cmd> <tau>
  local tag=$1 cmd=$2 tau=$3
  rm -rf $OUT/$tag
  env $P $cmd $BIN $COMMON $TL --out $OUT/$tag --sync grad --sync_every $tau \
      > $STAGE/$tag.log 2>&1
  rc=$?
  local tok wall ar tps
  tok=$(grep -o 'total *[0-9]*' $STAGE/$tag.log | tail -1 | awk '{print $2}')
  wall=$(grep -o 'wall_s *[0-9.]*' $STAGE/$tag.log | tail -1 | awk '{print $2}')
  ar=$(grep -o 'allreduce_s *[0-9.]*' $STAGE/$tag.log | tail -1 | awk '{print $2}')
  tps=$(awk -v t="${tok:-0}" -v w="${wall:-0}" 'BEGIN{ if (w>0) printf "%.1f", t/w; else print "?" }')
  printf '  %-12s rc=%s  tokens=%s  wall=%ss  tokens/s=%s  allreduce_total=%ss\n' \
      "$tag" "$rc" "$tok" "$wall" "$tps" "$ar"
  [ "$rc" != 0 ] && tail -4 $STAGE/$tag.log | sed 's/^/      /'
  echo -n "    allreduce por chamada (ms, ordenado): "
  grep 'sync step' $STAGE/$tag.log | grep -o 'allreduce_s *[0-9.]*' | awk '{printf "%.1f\n", 1000*$2}' | sort -n | paste -sd' ' -
  grep -h 'bit-identical' $STAGE/$tag.log | sed 's/^/    /'
  loads
  return 0
}
run_t 1r_tau1   "$MPL" 1
run_t 2r_tau1   "$MPX" 1
run_t 1r_tau100 "$MPL" 100
run_t 2r_tau100 "$MPX" 100

echo
echo "=== resumo ==="
for tag in 1r_tau1 2r_tau1 1r_tau100 2r_tau100; do
  [ -f $STAGE/$tag.log ] || continue
  tok=$(grep -o 'total *[0-9]*' $STAGE/$tag.log | tail -1 | awk '{print $2}')
  wall=$(grep -o 'wall_s *[0-9.]*' $STAGE/$tag.log | tail -1 | awk '{print $2}')
  ar=$(grep -o 'allreduce_s *[0-9.]*' $STAGE/$tag.log | tail -1 | awk '{print $2}')
  printf '  %-10s %8.1f tok/s (wall %ss, allreduce %ss)\n' "$tag" \
      "$(awk -v t="${tok:-0}" -v w="${wall:-1}" 'BEGIN{print t/w}')" "$wall" "$ar"
done
echo "### fim $(date '+%F %H:%M:%S')"
