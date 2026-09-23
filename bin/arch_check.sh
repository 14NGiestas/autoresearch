#!/usr/bin/env bash
# arch_check — o teste de concordancia FORTRAN <-> PYTHON da identidade da arch.
#
# Por que existe: a arch morava em tres lugares (os `parameter` de
# fortran_arch.f90, o __metadata__ do safetensors e o sidecar arch.txt) e o
# vinculo binario<->checkpoint era um NOME de pasta de build. A identidade
# canonica+id substitui isso -- mas so' serve se as duas implementacoes (Fortran
# e Python) concordarem BYTE A BYTE. Se discordarem, a checagem vira teatro.
#
# Uso: bin/arch_check.sh [DIR ...]     (default: /tmp/mix/init3m)
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DIRS=("$@")
[ ${#DIRS[@]} -eq 0 ] && DIRS=("/tmp/mix/init3m")

B=${ARCH_CHECK_BUILD:-/tmp/arch_check_build}
if [ ! -x "$(ls "$B"/*/app/arch_id 2>/dev/null | tail -1 || true)" ]; then
  echo "[arch_check] build em $B"
  nix develop "${NIX_SHELL:-.#cpu-only}" --command bash -c \
      "cd src && fortran-fpm build --build-dir '$B'" >/dev/null 2>&1 || {
        echo "[arch_check] build falhou; rode de novo sem redirecionar a saida"; exit 1; }
fi
BIN=$(ls "$B"/*/app/arch_id | tail -1)

echo "[arch_check] binario: $BIN"
"$BIN"
FORT_ID=$("$BIN" | awk '/^id /{print $2}')

fail=0
for d in "${DIRS[@]}"; do
  [ -d "$d" ] || { echo "  (pulado: $d nao existe)"; continue; }
  f=$("$BIN" --from "$d" 2>/dev/null)
  fid=$(echo "$f" | awk '/^id /{print $2}')
  fcan=$(echo "$f" | awk '/^canonical /{sub(/^canonical /,""); print}')
  p=$(/home/pauli/autoresearch/.venv-numpy/bin/python3 scripts/arch.py "$d" 2>/dev/null)
  pid=$(echo "$p" | awk '/^id /{print $2}')
  pcan=$(echo "$p" | awk '/^canonical /{sub(/^canonical /,""); print}')
  if [ "$fcan" = "$pcan" ] && [ "$fid" = "$pid" ]; then
    echo "  OK   $d  id=$fid"
  else
    echo "  FAIL $d"
    echo "       fortran: $fcan  ($fid)"
    echo "       python : $pcan  ($pid)"
    fail=1
  fi
done

# o binario compilado tem uma identidade propria; ela nao precisa casar com o
# checkpoint (pode ser outra arch de proposito) -- mas precisa ser estavel.
echo "[arch_check] id da arch compilada: $FORT_ID"
if [ "$fail" != 0 ]; then
  echo "[arch_check] FALHOU: as duas implementacoes discordam"
  exit 1
fi
echo "[arch_check] OK — Fortran e Python concordam na identidade da arch"
