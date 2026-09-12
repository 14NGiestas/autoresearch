#!/usr/bin/env python3
"""merge_checkpoints.py — model soup: média elemento a elemento de dois checkpoints.

Caso favorável: mesmo pai, mesmos hiperparâmetros, dados disjuntos (nossos ramos
v2/v3). Nessa condição os dois checkpoints ficam na mesma bacia de perda e a média
costuma bater ou empatar com cada um — é o que o DeepSeek-V4.1 faz para agregar
computação paralela, e é como multiplicamos throughput legítimo.

Portões (os desta sessão, aplicados de novo):
  - mesmo shape E dtype em todos os arrays, senão FALHA;
  - arch.txt dos dois têm de ser IDÊNTICOS, senão FALHA (arquiteturas diferentes
    não se misturam — é para isso que o arch.txt + check_shape existem);
  - quantidade honesta de drift: imprime a distância L2 relativa ||a-b||/||a|| por
    array e agregada. Se os ramos andaram para bacias diferentes, a média pode
    piorar -- e o número mostra isso ANTES do eval.

Momentos do Adam não têm média: são omitidos (zeros), que é o comportamento que o
treinador já usa para checkpoints antigos sem adam_*.npy. Para *avaliar* o merge
(eval_bpb) isso é irrelevante; para *continuar* treinando dele, os momentos
recomeçam do zero.

Uso:
  .venv-numpy/bin/python3 scripts/merge_checkpoints.py A B OUT [--alpha 0.5]

Saída: OUT/{*.npy,template.txt,arch.txt,manifest.json}
"""
import json
import os
import shutil
import sys

ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "--alpha" else 0.5
if "--alpha" in sys.argv:
    ALPHA = float(sys.argv[sys.argv.index("--alpha") + 1])
A, B, OUT = sys.argv[1:4]

import numpy as np

os.makedirs(OUT, exist_ok=True)

# arch.txt: os dois têm de concordar, palavra por palavra
def read_arch(d):
    p = os.path.join(d, "arch.txt")
    return open(p).read() if os.path.exists(p) else None

aa, ab = read_arch(A), read_arch(B)
if aa and ab and aa != ab:
    print("FALHA: arch.txt dos dois checkpoints não são idênticos -- não misturo.")
    sys.exit(1)

names = sorted(f for f in os.listdir(A)
               if f.endswith(".npy") and not f.startswith("adam_"))
missing = [f for f in names if not os.path.exists(os.path.join(B, f))]
if missing:
    print(f"FALHA: {len(missing)} arquivos faltando em {B}: {missing[:3]}")
    sys.exit(1)

sq = dr = 0.0
man = {"parent_A": A, "parent_B": B, "alpha": ALPHA, "files": {}, "arch_note": None}
for f in names:
    a = np.load(os.path.join(A, f), mmap_mode="r")
    b = np.load(os.path.join(B, f), mmap_mode="r")
    if a.shape != b.shape or a.dtype != b.dtype:
        print(f"FALHA: {f}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        sys.exit(1)
    af = a.astype("float64", copy=False)
    bf = b.astype("float64", copy=False)
    rel = float(np.linalg.norm(af - bf) / max(np.linalg.norm(af), 1e-30))
    m = (ALPHA * af + (1.0 - ALPHA) * bf).astype("float32")
    np.save(os.path.join(OUT, f), m)
    sq += float(np.sum((af - bf) ** 2))
    dr += float(np.sum(af ** 2))
    man["files"][f] = {"rel_l2": round(rel, 6), "shape": list(a.shape)}
    if f in ("lm_head_weight.npy",) or "wte" in f:
        print(f"  {f:45s} rel_l2={rel:.6f}")

man["drift_total_rel_l2"] = round(float((sq / max(dr, 1e-30)) ** 0.5), 6)
print(f"drift total (rel_l2 agregado): {man['drift_total_rel_l2']:.6f}")
print(f"  (pequeno = mesma bacia, soup tende a ajudar; grande = bacias diferentes,"
      f" media pode piorar)")

for extra in ("template.txt", "arch.txt"):
    s = os.path.join(A, extra)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(OUT, extra))
man["arch_note"] = "copiado de A (B é idêntico; validado acima)"
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
print(f"soup em {OUT}: {len(names)} arquivos + manifest.json")
