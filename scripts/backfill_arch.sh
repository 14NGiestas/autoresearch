#!/bin/bash
# backfill_arch.sh — escreve arch.txt nos checkpoints antigos (arquitetura 768/6/6/12).
# Confere a forma do wte.npy ANTES de declarar: declarar sem conferir seria repetir
# o pecado do dia. Checkpoint incompleto/parcial é PULADO, não adivinhado.
set -u
N=0; SKIP=0; WARN=0
read -r -d '' FILES <<'EOF' || true
d_model = 768
n_head = 6
n_kv = 6
n_layer = 12
vocab = 8192
ctx = 2048
bos = 8188
head_dim = 128
# backfill: pesos anteriores ao arch.txt; forma conferida contra wte.npy (8192x768)
# e n_layer contado pelos transformer_h_*.npy
EOF
for d in /tmp/w_* /tmp/w_*/*; do
  [ -d "$d" ] || continue
  ls "$d"/*.npy > /dev/null 2>&1 || continue
  [ -f "$d/arch.txt" ] && { SKIP=$((SKIP+1)); continue; }
  got=$(/usr/bin/python3 - "$d" 2>/dev/null <<'PY'
import glob, os, sys
try:
    import numpy as np
    d = sys.argv[1]
    w = [f for f in glob.glob(os.path.join(d, '*wte*.npy')) if 'adam' not in f]
    if not w:
        print("nowte"); raise SystemExit
    try:
        n = int(np.load(w[0], mmap_mode='r').size)
    except Exception:
        print("bad"); raise SystemExit
    L = len(glob.glob(os.path.join(d, 'transformer_h_*_attn_c_q_weight.npy')))
    print(f"{n} {L}")
except Exception:
    print("bad")
PY
)
  set -- ${got:-bad} ${2:-0}
  if [ "$1" != "6291456" ] || [ "${2:-0}" != "12" ]; then
    echo "    PULADO $d (wte=$1 camadas=${2:-?}) -- forma não confere ou incompleto"
    WARN=$((WARN+1)); continue
  fi
  printf '%s\n' "$FILES" > "$d/arch.txt"
  N=$((N+1))
done
echo "  arch.txt escrito em $N checkpoints ($SKIP já tinham, $WARN pulados)"
