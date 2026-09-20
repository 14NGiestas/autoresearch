#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.5.2",
# ]
# ///
"""ckio.py — I/O de checkpoint AGNOSTICO DE FORMATO (fase 2 do backend st).

Um checkpoint tem duas representacoes possiveis:

  * `st`  (padrao do trainer desde a fase 2): `model.safetensors` com nomes
    canonicos (`wte`, `lm`, `l{ll}.q/k/v/p/fc/p2`) + estado do otimizador em
    `.npy` (`adam_*`, `muon_*`) + `arch.txt`/`template.txt`;
  * `npy` (legado): `transformer_*.npy`/`lm_head_weight.npy` + o mesmo estado.

Este modulo e o ponto unico de leitura/escrita para o tooling Python. As chaves
publicas sao os NOMES LOGICOS = o nome do arquivo `.npy` correspondente
(ex.: `transformer_h_3_attn_c_q_weight.npy`), porque era assim que todas as
ferramentas ja pensavam; a traducao para nome canonico vive numa tabela unica em
`st_read.py` (`canon_of`/`npy_of`). Assim uma ferramenta que soma, escala ou
permuta tensores nao precisa saber de onde os bytes vieram.

Leitura de `st` usa `st_read.py` (oraculo pure-Python da biblioteca, sem numpy e
sem o pacote `safetensors`); `np.load` fica so no caminho legado e no estado.

Semantica que este modulo NAO muda (por ferramenta):
  * `fedavg_rounds.avg_dirs` media TAMBEM os momentos (`load_all`);
  * `merge_checkpoints` omite momentos (usa so `load_ckpt_dir`);
  * quem escreve (rescale/cirurgia/rebasin/merge/fedavg) escreve no MESMO
    formato da entrada (`like=`), porque um checkpoint st reescrito como .npy
    (ou vice-versa) quebraria o resto do pipeline sem avisar.

A guarda da fase 1 (`require_weights` abortando em st-only) virou aviso: com o
tooling migrado, st-only e um caso normal.
"""
import os
import sys

import numpy as np

import st_read

STATE_PREFIXES = ("adam_", "muon_")


# --------------------------------------------------------------------- formato
def has_st(d):
    return os.path.isfile(os.path.join(d, "model.safetensors"))


def st_path(d):
    return os.path.join(d, "model.safetensors")


def fmt(d):
    """'st' | 'npy' | 'both' | None (None = nenhum peso no diretorio)."""
    if not os.path.isdir(d):
        return None
    st = has_st(d)
    npys = bool(weight_files(d))
    if st and npys:
        return "both"
    if st:
        return "st"
    if npys:
        return "npy"
    return None


def weight_files(d):
    """Arquivos .npy de PESO (sem estado de otimizador) no layout legado."""
    if not os.path.isdir(d):
        return []
    return [f for f in sorted(os.listdir(d))
            if f.endswith(".npy") and not f.startswith(STATE_PREFIXES)]


def state_files(d):
    """Arquivos .npy de estado do otimizador (adam_/muon_) -- iguais nos 2 formatos."""
    if not os.path.isdir(d):
        return []
    return [f for f in sorted(os.listdir(d))
            if f.endswith(".npy") and f.startswith(STATE_PREFIXES)]


# --------------------------------------------------------------------- leitura
def load_ckpt_dir(d):
    """{nome_logico: ndarray} dos PESOS (nunca do estado do otimizador).

    `st`: le model.safetensors com o oraculo da biblioteca e remapeia canonico ->
    nome logico. `npy`: np.load. Um tensor canonico desconhecido (arquivo de
    outra ferramenta) e mantido com a propria chave canonica, para nao ser
    descartado em silencio.
    """
    if not os.path.isdir(d):
        raise SystemExit(f"ckio: {d} nao e um diretorio")
    if has_st(d):
        meta, tens = st_read.read(st_path(d))
        arrs = st_read.to_numpy(tens)
        out = {}
        for canon, a in arrs.items():
            logico = st_read.npy_of(canon) or canon
            if logico != canon and logico in out:
                raise SystemExit(f"ckio: {d}: dois tensores para {logico}")
            out[logico] = a
        if not out:
            raise SystemExit(f"ckio: {st_path(d)} nao tem tensor nenhum")
        return out
    files = weight_files(d)
    if not files:
        raise SystemExit(
            f"ckio: {d} nao tem peso nenhum (nem model.safetensors nem "
            f"transformer_*.npy)")
    return {f: np.load(os.path.join(d, f)) for f in files}


def load_state(d):
    """{nome_arquivo: ndarray} do estado do otimizador (adam_*, muon_*)."""
    return {f: np.load(os.path.join(d, f)) for f in state_files(d)}


def load_all(d):
    """Pesos + estado, para quem media os dois (fedavg)."""
    out = load_ckpt_dir(d)
    out.update(load_state(d))
    return out


def weight_names(d):
    """Nomes logicos dos pesos, SEM carregar os dados."""
    if has_st(d):
        meta, tens = st_read.read(st_path(d))
        return sorted((st_read.npy_of(c) or c) for c in tens)
    return weight_files(d)


def meta_of(d):
    """Metadata do model.safetensors ({} no legado)."""
    if not has_st(d):
        return {}
    meta, _ = st_read.read(st_path(d))
    return meta


def arch_of(d):
    """{chave: valor} do arch.txt ({} se nao houver)."""
    p = os.path.join(d, "arch.txt")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().split()[0]
    return out


# --------------------------------------------------------------------- escrita
def _st_tensors(weights):
    """[(canonico, dtype, shape, payload)] pronto para reference_writer.write."""
    out = []
    for nome, a in weights.items():
        a = np.asarray(a)
        canon = st_read.canon_of(nome) or nome
        dt = st_read.DTYPE_ST.get(str(a.dtype.newbyteorder("=")))
        if dt is None:
            raise SystemExit(f"ckio: dtype {a.dtype} de {nome} nao tem equivalente "
                             f"safetensors (esperado float32/float64/int*/uint8/bool)")
        if a.ndim > 1 and not a.flags["C_CONTIGUOUS"]:
            raise SystemExit(f"ckio: {nome} e {a.ndim}-D e nao-C-contiguo; achate "
                             f"(.ravel()) antes de gravar em safetensors")
        shape = list(a.shape)
        out.append((canon, dt, shape, np.ascontiguousarray(a).tobytes()))
    out.sort(key=lambda t: t[0])
    return out


def _write_st(d, weights, op=None, extra_meta=None, arch=None):
    rw, _ = st_read.find_oracle()
    if rw is None:
        raise SystemExit("ckio: nao achei tools/reference_writer.py da biblioteca "
                         "(clone local ou src/build/*/dependencies/safetensors)")
    meta = {"format_version": "1",
            "producer": "autoresearch/scripts/" + (op or os.path.basename(__file__)),
            "n_tensors": str(len(weights))}
    for k, v in (arch or {}).items():
        meta["arch." + k] = str(v)
    for k, v in (extra_meta or {}).items():
        meta[k] = str(v)
    rw.write(st_path(d), _st_tensors(weights), meta)
    return st_path(d)


def save_ckpt_dir(d, weights, state=None, like=None, op=None, extra_meta=None,
                  fmt_out=None):
    """Grava `weights` (+ `state`) em `d` no MESMO formato de `like`.

    `like=None` (ou um diretorio sem peso) grava em `st` (o padrao novo).
    Um diretorio com as duas representacoes (`both`) grava em `st`.
    `fmt_out` ('st'|'npy') forca o formato de saida -- para CONVERTER um
    checkpoint legado em st (ou o contrario) explicitamente.
    Copia arch.txt/template.txt de `like`, e arch.* vai para o metadata do
    model.safetensors (e o que permite abrir um checkpoint st-only sem arch.txt).
    """
    os.makedirs(d, exist_ok=True)
    src_fmt = fmt(like) if like else None
    out_fmt = fmt_out or ("npy" if src_fmt == "npy" else "st")
    if out_fmt not in ("st", "npy"):
        raise SystemExit(f"ckio: fmt_out invalido: {out_fmt}")
    arch = arch_of(like) if like else {}
    if out_fmt == "npy":
        for nome, a in weights.items():
            if st_read.canon_of(nome) is None and not nome.endswith(".npy"):
                raise SystemExit(f"ckio: {nome} nao tem nome .npy legado conhecido")
            np.save(os.path.join(d, nome), np.asfortranarray(a))
    else:
        _write_st(d, weights, op=op, extra_meta=extra_meta, arch=arch)
    for nome, a in (state or {}).items():
        np.save(os.path.join(d, nome), np.asfortranarray(a))
    if like:
        for extra in ("arch.txt", "template.txt"):
            p = os.path.join(like, extra)
            if os.path.exists(p):
                with open(p) as fh:
                    txt = fh.read()
                with open(os.path.join(d, extra), "w") as fh:
                    fh.write(txt)
    return out_fmt


# ------------------------------------------------------------------ compat
def require_weights(d, who="ckio"):
    """Fase 1: abortava em checkpoint st. Fase 2: AVISA e devolve os nomes.

    Mantida porque os scripts antigos ainda a chamam; o que ela garantia
    ("nao ler so o estado do otimizador e virar lixo") agora e garantido pelo
    proprio `load_ckpt_dir`, que so devolve PESOS nos dois formatos.
    """
    f = fmt(d)
    if f is None:
        raise SystemExit(f"{who}: nenhum tensor de peso em {d}")
    if f in ("st", "both"):
        print(f"{who}: {d} esta em safetensors ({f}) -- lendo via ckio", file=sys.stderr)
    return weight_names(d)
