#!/bin/bash
# set_arch.sh — muda o tamanho do modelo: valida, reescreve o módulo (UM arquivo),
# rebuilda. Sem pré-processador e sem duplicação: a mudança aparece no git diff.
#
#   scripts/set_arch.sh <d> <heads> <kv_heads> <layers> <vocab> <ctx>
#   scripts/set_arch.sh 96 6 2 12 8192 1024      # ~3M params
#   scripts/set_arch.sh 768 6 6 12 8192 2048     # histórico (97,5M)
set -eu
D=${1:?d}; H=${2:?heads}; KV=${3:?kv_heads}; L=${4:?layers}; V=${5:?vocab}; C=${6:?ctx}
fail() { echo "  RECUSADO: $1"; exit 1; }
[ "$D" -gt 0 ] && [ "$H" -gt 0 ] && [ "$L" -gt 0 ] && [ "$V" -gt 0 ] && [ "$C" -gt 0 ] || fail "dimensões têm de ser positivas"
[ $((D % H)) -eq 0 ] || fail "d=$D não é divisível por heads=$H (HD seria fracionário)"
[ "$KV" -ge 1 ] && [ "$KV" -le "$H" ] || fail "kv_heads=$KV fora de 1..heads=$H"
[ "$V" -gt 8188 ] || fail "vocab=$V precisa ser > BOS=8188"
F=/home/pauli/autoresearch/src/lib/fortran_arch.f90
sed -i -E "s/^(  integer, parameter :: D_MODEL = ).*/\1$D/; s/^(  integer, parameter :: N_HEAD = ).*/\1$H/; \
  s/^(  integer, parameter :: N_KV = ).*/\1$KV/; s/^(  integer, parameter :: N_LAYER = ).*/\1$L/; \
  s/^(  integer, parameter :: VV = ).*/\1$V/; s/^(  integer, parameter :: TT = ).*/\1$C/" "$F"
# verificação: se o sed não pegou, o script TEM de falhar (no-op silencioso é
# exatamente a classe de bug que estamos combatendo)
for pair in "D_MODEL = $D" "N_HEAD = $H" "N_KV = $KV" "N_LAYER = $L" "VV = $V" "TT = $C"; do
  grep -q "$pair" "$F" || fail "a reescrita não pegou \"$pair\" -- layout do módulo mudou?"
done
echo "  arquitetura: d=$D heads=$H kv=$KV layers=$L vocab=$V ctx=$C (HD=$((D/H)))"
grep -E 'parameter :: (D_MODEL|N_HEAD|N_KV|N_LAYER|VV|TT)' "$F" | sed 's/^/    /'
# UMA pasta de build por arquitetura: glob pegando a pasta errada foi um erro real
# (o require_arch pegou, mas nao deveria ser possivel errar). O binario passa a ter
# endereco deterministico: build/arch_d<D>_h<heads>_kv<kv>_l<layers>_v<vocab>_c<ctx>
BD="build/arch_d${D}_h${H}_kv${KV}_l${L}_v${V}_c${C}"
cd /home/pauli/autoresearch/src && nix develop .. --command /home/pauli/fortran-fpm build \
    --profile release --flag "-march=native -ffast-math" --build-dir "$BD" 2>&1 | tail -1 | sed 's/^/  /'
echo "  binario: src/$BD/*/app/train_run"
