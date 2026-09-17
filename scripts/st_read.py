#!/usr/bin/env python3
"""st_read.py -- le um checkpoint safetensors escrito pelo trainer e confere.

Independente do Fortran: usa o oraculo pure-Python da propria biblioteca
(tools/reference_writer.py), que valida o header inteiro (JSON, dtypes,
offsets, cobertura do buffer) sem numpy e sem o pacote `safetensors`.

Uso:
    scripts/st_read.py --ckpt-dir DIR [--compare-npy] [--quiet]

  --ckpt-dir DIR    diretorio com model.safetensors (e, com --compare-npy, os .npy)
  --npy-dir DIR     onde estao os .npy (default: o proprio --ckpt-dir); util para
                    comparar um checkpoint safetensors contra os .npy de outro dir
  --compare-npy     compara o payload de cada tensor canonico com o payload do
                    .npy correspondente (BYTES, nao valores) -- o .npy sozinho e
                    lido com struct, sem numpy
  --quiet           imprime so o resumo final

Saida (exit 0 = tudo consistente):
    st      : caminho, bytes, numero de tensores
    card    : steps/lr/tokens/rows_file/metrics parseados do JSON do metadata
    tensor  : nome dtype shape bytes fnv1a64   (hash do payload)
Com --compare-npy, uma linha por tensor com OK/DIFF contra o .npy.

Exit codes: 1 = arquivo/validacao/metadata/mismatch; 2 = erro de uso.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys

# .npy canonico -> nome canonico do tensor (mesmo mapa do adaptador da lib).
NPY_TO_CANON = {
    "transformer_wte_weight.npy": "wte",
    "lm_head_weight.npy": "lm",
}
for _slot, _canon in (("attn_c_q", "q"), ("attn_c_k", "k"), ("attn_c_v", "v"),
                      ("attn_c_proj", "p"), ("mlp_c_fc", "fc"), ("mlp_c_proj", "p2")):
    NPY_TO_CANON["transformer_h_{L}_%s_weight.npy" % _slot] = "l{L}." + _canon

REQUIRED_META = ("format_version", "producer", "n_tensors", "card")
REQUIRED_ARCH = ("arch.d_model", "arch.n_head", "arch.n_kv", "arch.n_layer",
                 "arch.vocab", "arch.ctx", "arch.head_dim")


def find_oracle():
    """Acha tools/reference_writer.py: clone local, dependencia do fpm, ou $SAFETENSORS_LIB."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    cands = []
    if os.environ.get("SAFETENSORS_LIB"):
        cands.append(os.path.join(os.environ["SAFETENSORS_LIB"], "tools", "reference_writer.py"))
    cands.append(os.path.join(repo, "safetensors-fortran", "tools", "reference_writer.py"))
    for pat in ("src/build/*/dependencies/safetensors/tools/reference_writer.py",
                "build/*/dependencies/safetensors/tools/reference_writer.py",
                "../src/build/*/dependencies/safetensors/tools/reference_writer.py"):
        cands.extend(sorted(glob.glob(os.path.join(repo, pat))))
    for c in cands:
        if os.path.exists(c):
            sys.path.insert(0, os.path.dirname(c))
            import reference_writer  # noqa: PLC0415
            return reference_writer, c
    print("st_read: nao achei tools/reference_writer.py (clone da lib, "
          "dependencia do fpm ou $SAFETENSORS_LIB)", file=sys.stderr)
    return None, None


def npy_payload(path):
    """Payload cru de um .npy (bytes apos o header) + o header como texto."""
    blob = open(path, "rb").read()
    if blob[:6] != b"\x93NUMPY":
        raise ValueError("%s: nao e um .npy (magic)" % path)
    major = blob[6]
    if major == 1:
        (hlen,) = struct.unpack("<H", blob[8:10])
        off = 10 + hlen
    else:
        (hlen,) = struct.unpack("<I", blob[8:12])
        off = 12 + hlen
    hdr = blob[10 if major == 1 else 12: off].decode("latin-1")
    if "'<f4'" not in hdr and "'f4'" not in hdr:
        raise ValueError("%s: dtype nao e float32: %s" % (path, hdr.strip()))
    return blob[off:], hdr.strip()


def canon_to_npy(ckpt_dir):
    """{nome canonico: caminho do .npy} para os arquivos que existem."""
    out = {}
    for fn in sorted(os.listdir(ckpt_dir)):
        if not fn.endswith(".npy"):
            continue
        for pat, canon in NPY_TO_CANON.items():
            if "{L}" in pat:
                for layer in range(0, 64):
                    if fn == pat.replace("{L}", str(layer)):
                        out[canon.replace("{L}", str(layer))] = os.path.join(ckpt_dir, fn)
            elif fn == pat:
                out[canon] = os.path.join(ckpt_dir, fn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--npy-dir", default="")
    ap.add_argument("--compare-npy", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rw, oracle_path = find_oracle()
    if rw is None:
        return 1
    st_path = os.path.join(args.ckpt_dir, "model.safetensors")
    if not os.path.exists(st_path):
        print("st_read: %s nao existe" % st_path, file=sys.stderr)
        return 1

    problems = []
    try:
        meta, tensors = rw.read(st_path)
    except Exception as exc:  # noqa: BLE001
        print("st_read: FAIL header/validacao: %s" % exc, file=sys.stderr)
        return 1

    if not args.quiet:
        print("oracle  : %s" % oracle_path)
        print("st      : %s (%d bytes, %d tensores, %d chaves de metadata)"
              % (st_path, os.path.getsize(st_path), len(tensors), len(meta)))

    # ---- metadata obrigatoria
    for k in REQUIRED_META + REQUIRED_ARCH:
        if k not in meta:
            problems.append("metadata ausente: %s" % k)
    if meta.get("n_tensors") is not None:
        try:
            if int(meta["n_tensors"]) != len(tensors):
                problems.append("n_tensors=%s mas ha %d tensores"
                                % (meta["n_tensors"], len(tensors)))
        except ValueError:
            problems.append("n_tensors nao e inteiro: %r" % meta["n_tensors"])
    card = {}
    if "card" in meta:
        try:
            card = json.loads(meta["card"])
        except Exception as exc:  # noqa: BLE001
            problems.append("card nao e JSON valido: %s" % exc)
        else:
            for k in ("steps", "lr", "tokens", "rows_file", "metrics"):
                if k not in card:
                    problems.append("card sem a chave %s" % k)
            if not args.quiet:
                print("card    : steps=%s lr=%s tokens=%s rows_file=%s metrics=%s"
                      % (card.get("steps"), card.get("lr"), card.get("tokens"),
                         card.get("rows_file"), card.get("metrics")))
    if not args.quiet:
        print("arch    : d=%s heads=%s kv=%s layers=%s vocab=%s ctx=%s head_dim=%s"
              % (meta.get("arch.d_model"), meta.get("arch.n_head"), meta.get("arch.n_kv"),
                 meta.get("arch.n_layer"), meta.get("arch.vocab"), meta.get("arch.ctx"),
                 meta.get("arch.head_dim")))

    # ---- payloads + hashes
    npy_dir = args.npy_dir or args.ckpt_dir
    canon = canon_to_npy(npy_dir) if args.compare_npy else {}
    for name, info in tensors.items():
        h = rw.fnv1a64(info["data"])
        line = "%-10s %-4s %-14s %9d  fnv1a64=%016x" % (
            name, info["dtype"], str(info["shape"]), len(info["data"]), h)
        if args.compare_npy:
            src = canon.get(name)
            if src is None:
                line += "  DIFF: sem .npy correspondente"
                problems.append("%s: sem .npy correspondente" % name)
            else:
                payload, _hdr = npy_payload(src)
                if payload == info["data"]:
                    line += "  .npy OK (%s)" % os.path.basename(src)
                else:
                    line += "  DIFF contra %s (%d vs %d bytes)" % (
                        os.path.basename(src), len(info["data"]), len(payload))
                    problems.append("%s: bytes diferentes do .npy" % name)
        if not args.quiet:
            print("tensor  : %s" % line)

    if args.compare_npy:
        extra = set(canon) - set(tensors)
        if extra:
            problems.append(".npy sem tensor no .st: %s" % sorted(extra))
        if not args.quiet:
            print("compare : %d tensores, %d .npy" % (len(tensors), len(canon)))

    if problems:
        print("st_read: FAIL")
        for p in problems:
            print("  - %s" % p)
        return 1
    print("st_read: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
