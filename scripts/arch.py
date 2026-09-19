"""arch.py — a identidade da arquitetura, em Python, IGUAL ao Fortran.

Por que existe: a arch morava em tres lugares (os `parameter` de fortran_arch.f90,
o __metadata__ do safetensors e o sidecar arch.txt) e o vinculo
binario<->checkpoint era um NOME DE PASTA. Este modulo da UMA definicao da
identidade, compartilhada com o Fortran (arch_canonical_of/arch_id_of) e coberta
por teste de concordancia (bin/arch_check.sh).

Uso:
    python3 scripts/arch.py DIR              # arch, canonical e id de um checkpoint
    python3 scripts/arch.py --id-of JSON     # id de uma string canonica
    python3 scripts/arch.py --canonical-of d=96,nh=6,nkv=2,hd=16,nl=12,vv=8192,ctx=1024,bos=8188

O id NAO e' criptografico: rotacao+xor sobre os bytes da canonica. Escolhido
porque nao depende de overflow de inteiro assinado (norma nao define) e tem
implementacao identica nos dois lados. Ele so' precisa nao colidir entre configs.
"""

import json
import os
import struct
import sys

SCHEMA = "fortran_gpt/arch/1"
KEYS = ("d_model", "n_head", "n_kv", "head_dim", "n_layer", "vocab", "ctx", "bos")
MASK64 = (1 << 64) - 1


def canonical_of(d_model, n_head, n_kv, head_dim, n_layer, vocab, ctx, bos):
    """String canonica: ordem fixa, sem espaco. Tem de ser byte-igual ao Fortran."""
    return (
        '{"schema":"%s"' % SCHEMA
        + ',"d_model":%d' % d_model
        + ',"n_head":%d' % n_head
        + ',"n_kv":%d' % n_kv
        + ',"head_dim":%d' % head_dim
        + ',"n_layer":%d' % n_layer
        + ',"vocab":%d' % vocab
        + ',"ctx":%d' % ctx
        + ',"bos":%d' % bos
        + "}"
    )


def canonical(arch):
    """arch: dict com as KEYS."""
    if tuple(arch) != KEYS and set(arch) != set(KEYS):
        raise ValueError("arch nao tem exatamente as chaves esperadas: %s" % (KEYS,))
    return canonical_of(*[arch[k] for k in KEYS])


def arch_id_of(s):
    """16 digitos hex. Identica a arch_id_of do fortran_arch_mod."""
    acc = 0x0123456789ABCDEF
    for b in s.encode():
        acc ^= b
        acc = ((acc << 17) | (acc >> 47)) & MASK64
        acc ^= acc >> 29
    return "%016x" % acc


def arch_id(arch):
    return arch_id_of(canonical(arch))


def read_arch(ckpt_dir):
    """arch de um checkpoint: __metadata__ do safetensors PRIMEIRO (padrao, viaja
    com os pesos), arch.txt como fallback (iniciativas antigas / init em .npy).
    Devolve (arch_dict, fonte). Falha alto se nao achar nem um nem outro."""
    ckpt_dir = os.path.expanduser(ckpt_dir)
    for name in sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []:
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(ckpt_dir, name)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n <= 0 or n > 100_000_000:
                raise ValueError("header do safetensors invalido em %s" % path)
            header = json.loads(fh.read(n))
        meta = header.get("__metadata__") or {}
        if all("arch.%s" % k in meta for k in KEYS):
            return {k: int(meta["arch.%s" % k]) for k in KEYS}, path
    txt = os.path.join(ckpt_dir, "arch.txt")
    if os.path.exists(txt):
        arch = {}
        for line in open(txt):
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k in KEYS:
                arch[k] = int(v.strip().split()[0])
        if set(arch) == set(KEYS):
            return arch, txt
        raise ValueError("arch.txt incompleto em %s" % txt)
    raise SystemExit("arch.py: sem .safetensors com metadata nem arch.txt em %s" % ckpt_dir)


def main(argv):
    if len(argv) >= 3 and argv[1] == "--id-of":
        print(arch_id_of(argv[2]))
        return 0
    if len(argv) >= 3 and argv[1] == "--canonical-of":
        kv = dict(p.split("=") for p in argv[2].split(","))
        print(canonical({k: int(kv[k]) for k in KEYS}))
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 2
    arch, src = read_arch(argv[1])
    print("canonical %s" % canonical(arch))
    print("id %s" % arch_id(arch))
    print("# fonte: %s" % src)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
