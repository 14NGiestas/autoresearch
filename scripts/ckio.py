#!/usr/bin/env python3
"""ckio.py — guarda de formato para o tooling que ainda le .npy.

O trainer passou a gravar o padrao em safetensors (`--ckpt-format st`, default
novo): um checkpoint novo tem `model.safetensors` + estado do otimizador
(`adam_*.npy`) + arch.txt/template.txt, e NAO tem mais os `transformer_*.npy` de
peso. Ferramentas que varrem `.npy` (merge, media, cirurgia, reescala) passariam
a ver apenas o estado do otimizador -- e produziriam lixo silenciosamente.

Esta guarda troca o lixo silencioso por um erro explicito. Quando os
consumidores migrarem para scripts/st_read.py (fase 2), a guarda sai.
"""
import os

SKIP = ("adam_", "muon_")


def weight_npys(d):
    """Nomes dos .npy de PESO no diretorio (sem estado de otimizador)."""
    if not os.path.isdir(d):
        raise SystemExit(f"ckio: {d} nao e um diretorio")
    return [f for f in sorted(os.listdir(d))
            if f.endswith(".npy") and not f.startswith(SKIP)]


def require_weights(d, who="ckio"):
    """Devolve os .npy de peso ou aborta com a razao (st-only / vazio)."""
    w = weight_npys(d)
    if w:
        return w
    st = os.path.join(d, "model.safetensors")
    if os.path.exists(st):
        raise SystemExit(
            f"{who}: {d} e um checkpoint safetensors (st-only, o padrao novo).\n"
            f"  Este script ainda le o layout .npy. Use scripts/st_read.py para "
            f"inspecionar/converter,\n  ou rode o trainer com --ckpt-format npy "
            f"para um checkpoint legado. (migracao dos consumidores = fase 2)")
    raise SystemExit(f"{who}: nenhum tensor de peso em {d}")
