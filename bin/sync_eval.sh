#!/usr/bin/env bash
# bin/sync_eval.sh — envia um BUNDLE PORTABLE (binário + closure + wrapper) para
# uma caixa remota, com arquitetura explícita e verificada.
#
# Por que assim, e não mandando a árvore de código:
#   - a máquina remota (halfbeast) tem nix 2.6.0, velho demais para o nosso flake,
#     então "compile no destino" NÃO é uma opção. A versão anterior deste script
#     mandava o código e sugeria compilar lá -- uma instrução que não pode ser
#     seguida, e que já custou uma sessão de depuração.
#   - o binário é -march=native: ele NÃO roda em outra microarquitetura sem a
#     closure de .so e o loader. Então viaja o binário + as libs + um wrapper.
#
# Regra de ouro (aprendida hoje): NADA de default silencioso de arquitetura. Se o
# build dir não for dito explicitamente, ele é derivado de src/lib/fortran_arch.f90
# (fonte de verdade única) e IMPRESSO; se não existir, o script PARA e lista o que
# existe. Enviar o binário de 768 onde se espera o de 96 é exatamente a classe de
# bug que a sessão toda combateu.
#
# Uso:
#   bin/sync_eval.sh                                   # deriva arquitetura do módulo
#   bin/sync_eval.sh --build-dir build/arch_d96_...    # explícito
#   bin/sync_eval.sh --apps "train_run eval_bpb" --ckpt /tmp/mix/init3m \
#                    --rows /tmp/mix/rows_f75.txt --dest autoresearch-eval
#   bin/sync_eval.sh --verify                          # roda --help remoto (srun -c 1)
set -euo pipefail
HOST="${HOST:-halfbeast}"
DEST="${DEST:-autoresearch-eval}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR=""
APPS=""
CKPT=""
ROWS=""
VERIFY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --apps) APPS="$2"; shift 2;;
    --ckpt) CKPT="$2"; shift 2;;
    --rows) ROWS="$2"; shift 2;;
    --verify) VERIFY=1; shift;;
    -h|--help) sed -n '2,25p' "$0"; exit 0;;
    *) echo "argumento desconhecido: $1 (--help)"; exit 2;;
  esac
done

# ---- arquitetura: do módulo (fonte de verdade única) --------------------------
ARCH_MOD="$SRC/src/lib/fortran_arch.f90"
[ -f "$ARCH_MOD" ] || { echo "não achei $ARCH_MOD"; exit 1; }
geti() { grep -oE "integer, parameter :: $1 = [0-9]+" "$ARCH_MOD" | grep -oE '[0-9]+$'; }
D=$(geti D_MODEL); H=$(geti N_HEAD); KV=$(geti N_KV); L=$(geti N_LAYER)
V=$(geti VV); C=$(geti TT)
[ -n "$D" ] || { echo "não consegui ler a arquitetura de $ARCH_MOD"; exit 1; }
ARCH="d${D}_h${H}_kv${KV}_l${L}_v${V}_c${C}"
if [ -z "$BUILD_DIR" ]; then
  BUILD_DIR="src/build/arch_${ARCH}"
  echo "== arquitetura derivada do MÓDULO: d=$D heads=$H kv=$KV layers=$L vocab=$V ctx=$C"
else
  # A pasta de build É a declaração do que foi compilado. Se o módulo divergir,
  # o módulo foi alterado DEPOIS do build -- e é a pasta que manda no manifesto,
  # senão o manifesto mente sobre o que está viajando (classe de bug do dia).
  BD_ARCH=$(basename "$BUILD_DIR")
  echo "== arquitetura da PASTA de build: $BD_ARCH"
  for kv in D_MODEL:$D N_HEAD:$H N_KV:$KV N_LAYER:$L VV:$V TT:$C; do
    nm=${kv%%:*}; val=${kv#*:}
    case "$nm" in
      D_MODEL) pat="_d${val}_";; N_HEAD) pat="_h${val}_";; N_KV) pat="_kv${val}_";;
      N_LAYER) pat="_l${val}_";; VV) pat="_v${val}_";; TT) pat="_c${val}$";;
    esac
    echo "$BD_ARCH" | grep -q "$pat" || {
      echo "  AVISO: módulo diz $nm=$val e a pasta diz outra coisa -> usando a PASTA"
      break; }
  done
  # re-deriva os valores do manifesto a partir do nome da pasta
  set -- $(echo "$BD_ARCH" | sed -E 's/arch_d([0-9]+)_h([0-9]+)_kv([0-9]+)_l([0-9]+)_v([0-9]+)_c([0-9]+)/\1 \2 \3 \4 \5 \6/')
  if [ $# -eq 6 ]; then D=$1; H=$2; KV=$3; L=$4; V=$5; C=$6; ARCH="$BD_ARCH"; fi
fi
echo "== build dir: $BUILD_DIR"

if [ ! -d "$SRC/$BUILD_DIR" ]; then
  echo "ERRO: $BUILD_DIR não existe. Builds disponíveis:"
  ls -d "$SRC"/src/build/arch_* 2>/dev/null | sed 's|.*/build/|  |' || echo "  (nenhum)"
  echo "  Gere um com: scripts/set_arch.sh $D $H $KV $L $V $C"
  exit 1
fi
BIN_DIR="$(ls -d "$SRC/$BUILD_DIR"/*/app 2>/dev/null | head -1)"
[ -n "$BIN_DIR" ] || { echo "ERRO: nenhum app/ em $BUILD_DIR"; exit 1; }
[ -n "$APPS" ] || APPS="train_run eval_bpb chat_text repl"
for a in $APPS; do
  [ -x "$BIN_DIR/$a" ] || { echo "ERRO: binário $a não existe em $BIN_DIR"; exit 1; }
done
echo "== apps: $APPS   (de $BIN_DIR)"

# ---- closure de bibliotecas ---------------------------------------------------
# O binário é -march=native e linkado contra o nix: ele precisa do loader e das
# .so exatas. ldd dá os caminhos absolutos; achatamos em lib/ (o wrapper usa
# --library-path, então uma pasta só basta).
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/lib" "$TMP/bin"
LOADER=""
for a in $APPS; do cp "$BIN_DIR/$a" "$TMP/bin/"; done
LIBS="$(ldd $BIN_DIR/$(echo $APPS | awk '{print $1}') | awk '$2=="=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}' | sort -u)"
for l in $LIBS; do
  case "$(basename "$l")" in
    ld-linux-x86-64.so.2) LOADER="$l";;
    *) cp -f "$l" "$TMP/lib/" 2>/dev/null || true;;
  esac
done
[ -n "$LOADER" ] || LOADER="$(ls /nix/store/*glibc*/lib/ld-linux-x86-64.so.2 2>/dev/null | head -1)"
[ -n "$LOADER" ] && cp -f "$LOADER" "$TMP/lib/" || { echo "ERRO: não achei o loader"; exit 1; }
NLIBS=$(ls "$TMP/lib" | wc -l)
echo "== closure: $NLIBS bibliotecas + loader"

# ---- wrapper + manifesto ------------------------------------------------------
for a in $APPS; do
  cat > "$TMP/run_$a.sh" <<EOF
#!/bin/sh
# Roda o $a compilado na fermi usando as libs enviadas junto (mesmo binário e
# mesma OpenBLAS da caixa de treino: só a CPU difere). Sem /nix, sem sudo.
set -u
D=\$(cd "\$(dirname "\$0")" && pwd)
exec "\$D/lib/ld-linux-x86-64.so.2" --library-path "\$D/lib" "\$D/bin/$a" "\$@"
EOF
  chmod +x "$TMP/run_$a.sh"
done
{
  echo "# bundle portable enviado por bin/sync_eval.sh"
  echo "data = $(date -Is)"
  echo "origem = $(hostname -s)"
  echo "build_dir = $BUILD_DIR"
  echo "arch = $ARCH"
  echo "d_model = $D"
  echo "n_head = $H"
  echo "n_kv = $KV"
  echo "n_layer = $L"
  echo "vocab = $V"
  echo "ctx = $C"
  echo "apps = $APPS"
  echo "libs = $NLIBS"
  for a in $APPS; do
    printf 'sha256_%s = %s\n' "$a" "$(sha256sum "$TMP/bin/$a" | cut -c1-64)"
  done
} > "$TMP/manifest.txt"

# ---- envio --------------------------------------------------------------------
echo "== enviando para $HOST:$DEST/portable"
ssh "$HOST" "mkdir -p $DEST/portable"
rsync -a --delete --exclude 'slurm-*.out' "$TMP/" "$HOST:$DEST/portable/"
rsync -a "$HOME/.cache/autoresearch/tok_tables/" "$HOST:$DEST/tok_tables/"
if [ -n "$CKPT" ]; then
  [ -f "$CKPT/arch.txt" ] || echo "  AVISO: $CKPT não tem arch.txt (require_arch não vai validar)"
  rsync -a "$CKPT/" "$HOST:$DEST/$(basename "$CKPT")/"
  echo "== checkpoint $(basename "$CKPT") enviado"
fi
if [ -n "$ROWS" ]; then
  rsync -a "$ROWS" "$HOST:$DEST/$(basename "$ROWS")"
  echo "== rows $(basename "$ROWS") enviado"
fi

# ---- verificação: o binário EXECUTA lá? ---------------------------------------
if [ "$VERIFY" = 1 ]; then
  A=$(echo $APPS | awk '{print $1}')
  echo "== verificando execução remota (srun -c 1, educado):"
  ssh "$HOST" "cd $DEST/portable && srun -p compute -c 1 --time 00:02:00 sh -c './run_$A.sh --help' 2>&1 | head -3" || true
fi

cat <<EOF

== pronto. Para rodar como job (cheque a carga antes: bin/hb.sh status):
   ssh $HOST 'cd $DEST/autoresearch && sbatch -p compute -c 8 --mem 16G --wrap "\\
     OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE \\
     $DEST/portable/run_train_run.sh --weights \\\$HOME/$DEST/<ckpt> --rows \\\$HOME/$DEST/<rows> \\
       --out /tmp/<out> --nsteps N --ntrain M --nval 5 --val_every 1000 --attn blas \\
       --bytes \\\$HOME/$DEST/tok_tables/token_bytes.txt"'
   # o manifesto em $DEST/portable/manifest.txt diz qual arquitetura está lá;
   # o require_arch aborta se o checkpoint não bater com ela.
EOF
