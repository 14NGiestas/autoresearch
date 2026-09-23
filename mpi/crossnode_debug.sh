#!/bin/bash
# mpi/crossnode_debug.sh -- DIAGNOSTICO do lancamento remoto do hydra (fermi+halfbeast).
#
# Sintoma: `mpirun -f /tmp/ddp/hosts -ppn 1 -n 2 hostname` falha com
# "Launch proxy failed / callback returned error status" -- ssh e o proxy
# funcionam separadamente (verificado pelo supervisor). Este script colhe o
# comando EXATO que o hydra usa para lancar o proxy remoto (HYDRA_LAUNCHER_EXTRA_ARGS=-v
# + -verbose) e testa variantes, para separar: (i) opcao de ssh nao suportada,
# (ii) ambiente/junk no shell remoto, (iii) caminho do proxy.
#
# SEGURANCA: TODOS os mpirun daqui usam -envlist EXPLICITO. Ambiente herdado faz
# o hydra encaminhar TUDO (token de API, chave de terminal) para o no remoto --
# regra de seguranca, nao preferencia. Ver docs/mpi_ddp.md secao 10.
#
# So' roda dentro do Slurm (o wrapper jobs/fermi_crossdbg.sbatch chama isto).
set -u
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "RECUSADO: so' dentro do Slurm (jobs/fermi_crossdbg.sbatch)."; exit 2
fi
cd "$(dirname "$0")/.."
STAGE=${STAGE:-/tmp/ddp}
HOSTS=$STAGE/hosts
ENV_FWD="PATH,LD_LIBRARY_PATH,HOME,TMPDIR,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS"
MP="mpirun -f $HOSTS -ppn 1 -n 2 -envlist $ENV_FWD"
SSH="ssh -o BatchMode=yes"

echo "### host $(hostname) $(date '+%F %H:%M:%S') job ${SLURM_JOB_ID}"
echo "### hosts: $(tr '\n' ' ' < $HOSTS)"
echo "### mpirun: $(command -v mpirun)  (hydra_pmi_proxy: $(dirname "$(command -v mpirun)")/hydra_pmi_proxy)"
echo "### ssh: $(command -v ssh)"
echo "### load: $(uptime | sed 's/.*load average/load/')  |  remoto: $($SSH halfbeast 'uptime | sed "s/.*load average/load/"')"

echo
echo "=== V0: ssh puro (baseline) ==="
$SSH halfbeast 'hostname; echo PATH=$PATH | cut -c1-120; which hostname' 2>&1 | sed 's/^/    /'

echo
echo "=== V1: mpirun trivial (deve FALHAR -- e' o sintoma) ==="
timeout 90 $MP hostname 2>&1 | tail -12 | sed 's/^/    /'
echo "    rc=$?"

echo
echo "=== V2: mesmo + ssh -v (comando exato do hydra) ==="
HYDRA_LAUNCHER_EXTRA_ARGS='-v' timeout 90 $MP -verbose hostname 2>&1 | tail -40 | sed 's/^/    /'

echo
echo "=== V3: -launcher ssh explicito + HYDRA_LAUNCHER=/usr/bin/ssh ==="
HYDRA_LAUNCHER=/usr/bin/ssh timeout 90 $MP -launcher ssh hostname 2>&1 | tail -8 | sed 's/^/    /'

echo
echo "=== V4: ssh sem tty e sem X11 (-T -x -o BatchMode=yes) ==="
HYDRA_LAUNCHER_EXTRA_ARGS='-T -x -o BatchMode=yes' timeout 90 $MP hostname 2>&1 | tail -8 | sed 's/^/    /'

echo
echo "=== V5: proxy lancado A MAO via ssh (isola o proxy do hydra) ==="
PROXY=$(dirname "$(command -v mpirun)")/hydra_pmi_proxy
$SSH halfbeast "$PROXY --version" 2>&1 | head -3 | sed 's/^/    /'
$SSH halfbeast "cd $STAGE && OMP_NUM_THREADS=1 $PROXY -h 2>&1 | head -3" | sed 's/^/    /'

echo
echo "=== V6: 2 ranks SO' na fermi pelo hostfile (hostfile/ppn ok?) ==="
sed 's/^halfbeast/#halfbeast/' $HOSTS > $STAGE/hosts_local
timeout 90 mpirun -f $STAGE/hosts_local -ppn 1 -n 2 -envlist $ENV_FWD hostname 2>&1 | tail -6 | sed 's/^/    /'

echo
echo "=== V7: rede fermi->halfbeast (o que o tau vai enfrentar) ==="
echo -n "    RTT (3x ssh true): "; for _ in 1 2 3; do /usr/bin/time -f "%es" $SSH halfbeast true 2>&1 | tail -1 | tr '\n' ' '; done; echo
echo -n "    50 MB por ssh (dd | pv): "; dd if=/dev/zero bs=1M count=50 2>/dev/null | /usr/bin/time -f "%es (%e s)" $SSH halfbeast 'cat > /dev/null' 2>&1 | tail -1
echo "    (comparar com 1 GbE = 125 MB/s -> 50 MB = 0,4 s)"
echo "### fim $(date '+%F %H:%M:%S')"
