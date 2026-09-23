#!/bin/bash
# mpi/crossnode_stage.sh -- staging do cross-node em /tmp/ddp NOS DOIS NOS.
#
# Excecao consciente a regra "nada em /tmp" (autorizada pelo supervisor para o
# cross-node): /tmp/ddp leva SO' o launcher -- o binario train_ddp, o npy de
# rows, a pasta init/ (pesos + adam do checkpoint de partida) e o hostfile. O
# checkpoint do run sai no no do rank 0 (fermi), nao em /tmp/ddp.
#
# Por que precisa: rank 1 roda no halfbeast, entao binario, rows e init tem de
# existir LA' (o mpirun local le os do rank 0). As bibliotecas vem do /nix/store
# dos dois lados (nix copy) -- o caminho tem de ser o MESMO nos dois, senao o
# hydra nao acha o hydra_pmi_proxy no no remoto.
set -eu
REMOTE=${REMOTE:-halfbeast}
STAGE=${STAGE:-/tmp/ddp}
cd "$(dirname "$0")/.."
BIN=$(ls -t mpi/build/*/app/train_ddp | head -1)
ROWS=$STAGE/rows.npy
INIT=${INIT:-/tmp/w_p2soup}
PYBIN=/home/pauli/autoresearch/.venv-numpy/bin/python3
SSH="ssh -o BatchMode=yes"
[ -d "$INIT" ] || { echo "ABORTA: $INIT nao existe"; exit 3; }
[ -x "$PYBIN" ] || { echo "ABORTA: $PYBIN nao existe (python com numpy)"; exit 3; }

# rows: 64 linhas de treino + 300 de holdout no MESMO arquivo (o app le treino em
# [start_row, start_row+ntrain) e val em [start_row+ntrain, +nval)). GERADO AQUI,
# nao herdado de scratch de outro teste -- foi exatamente isso que abortou o job
# 123 (dependia de mpi/scratch_*/rows_quant.npy, que ja' nao existia).
if [ ! -s "$ROWS" ]; then
  echo "### 0/5 gerando rows ($ROWS: 64 treino + 300 holdout)"
  "$PYBIN" - "$ROWS" <<'PYEOF' || { echo "ABORTA: falhou ao gerar rows"; exit 3; }
import sys
import numpy as np
tr = np.load('/tmp/mix/rows_dev.npy')[:64]        # 64 x 1025
va = np.load('/tmp/prose65/rows_val.npy')[:300]   # 300 x 1025
out = np.concatenate([tr, va]).astype(np.int32)
np.save(sys.argv[1], out)
print('    rows:', out.shape)
PYEOF
fi
[ -s "$ROWS" ] || { echo "ABORTA: $ROWS vazio"; exit 3; }

mkdir -p "$STAGE"
cp -f "$BIN" "$STAGE/train_ddp"
printf 'fermi:8\n%s:8\n' "$REMOTE" > "$STAGE/hosts"

echo "### 2/5 sonda de allreduce (mpif90, caminho de DADOS) + copias"
mpif90 -O2 -o "$STAGE/allreduce_probe" mpi/allreduce_probe.f90 || { echo "ABORTA: compile do probe"; exit 3; }
cp -f mpi/crossnode_sshwrap.sh mpi/crossnode_setup.sh "$STAGE/"
cp -f "$HOME/.cache/autoresearch/tok_tables/token_bytes.txt" "$STAGE/token_bytes.txt"

echo "### 1b/5 closure de bibliotecas (train_ddp + sonda) -> $REMOTE"
LIBS=$( { ldd "$BIN"; ldd "$STAGE/allreduce_probe"; } | grep -oE '/nix/store/[a-zA-Z0-9._+-]+' | sort -u )
echo "    $(echo "$LIBS" | wc -l) store paths"
nix copy --to "ssh://$REMOTE" $LIBS


echo "### 3/5 rows + init/ + binario + probe + wrappers + hosts -> $REMOTE:$STAGE"
tar -C "$(dirname "$INIT")" -cf - "$(basename "$INIT")" | \
  $SSH "$REMOTE" "mkdir -p $STAGE && tar -C $STAGE -xf -"
tar -C "$STAGE" -cf - train_ddp allreduce_probe crossnode_sshwrap.sh crossnode_setup.sh token_bytes.txt rows.npy hosts | \
  $SSH "$REMOTE" "mkdir -p $STAGE && tar -C $STAGE -xf - && chmod +x train_ddp allreduce_probe crossnode_sshwrap.sh"

echo "### 4/5 verificacao: binario e sonda RODAM no no remoto?"
$SSH "$REMOTE" "cd $STAGE && ./train_ddp --help >/dev/null 2>&1 && echo '    REMOTO: train_ddp --help OK' || echo '    REMOTO: train_ddp FALHOU'"
$SSH "$REMOTE" "cd $STAGE && ./allreduce_probe >/dev/null 2>&1; echo \"    REMOTO: probe sozinho exit=\$? (1 rank no halfbeast, esperado 0)\""
$SSH "$REMOTE" "ls $STAGE | tr '\n' ' '; echo; ls $STAGE/init/*.npy | wc -l | sed 's/^/    init: /;s/$/ .npy/'"

echo "### 5/5 o proxy do hydra existe no remoto (mesmo caminho do local)?"
PROXY=$(dirname "$(command -v mpirun)")/hydra_pmi_proxy
echo "    local : $PROXY"
$SSH "$REMOTE" "[ -x $PROXY ] && echo '    remoto: idem (executavel)' || echo '    remoto: AUSENTE'"
echo "### staging ok  $(date '+%F %H:%M:%S')"
