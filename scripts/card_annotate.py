#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""card_annotate.py — junta a CARACTERIZACAO ao arquivo de pesos.

O trainer escreve no __metadata__ o que ele sabe (arch, steps, lr, tokens,
rows-file). O que ele NAO sabe: energia (medida de fora, pelo wrapper, e so
depois que o job termina), metricas de eval (bpb/H2/ECE, medidas depois) e
linhagem (quem e o pai). Este script faz a juncao: le a linha do ledger de
energia, recebe as metricas na linha de comando, e REESCREVE o header do
model.safetensors com os campos `energy.*`, `eval.*` e `lineage.*` — o card
passa a viajar dentro do arquivo de pesos, que era o motivo de existir.

Precedencia de energia: se o card JA tem energy.self_measured=1 (o train_run
mediu a propria energia com o pacote fortran_energy e gravou o delta exato do
checkpoint), este script NAO sobrescreve NENHUM campo energy.* -- so preenche os
que faltam (source/attribution/J_per_Mtok). O rateio linear abaixo e o caminho
para checkpoints antigos, medidos de fora.

Atribuicao de energia (explicita, porque e uma escolha): a energia e medida por
JOB. O checkpoint recebe J * (passos_do_checkpoint / passos_totais_do_job).
A potencia e ~constante durante o job (medido: W_mean 40.9 W, pico 62 W no
arranque), entao o rateio linear erra ~1-2%. Sem --steps/--total-steps o J do job
inteiro vai para o checkpoint e o campo `attribution` diz isso.

Hashes: reescrever o header MUDA os bytes do arquivo. Anote ANTES de registrar o
checkpoint no ckpt.py (que hasheia o manifesto), senao o registro fica invalido.
O script confere que os PAYLOADS dos tensores nao mudaram (sha256 por tensor,
antes e depois) e grava um .bak.

Uso:
  scripts/card_annotate.py --ckpt DIR --tag cal/d96_8t --steps 430 --total-steps 430 \
      --eval bpb=2.12 --eval H2=0.11 --lineage parent=ckpt_abc --note "grid disj K=4"
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def oracle():
    p = os.path.join(REPO, "safetensors-fortran", "tools", "reference_writer.py")
    spec = importlib.util.spec_from_file_location("refw", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def payload_hashes(tensors):
    """tensors: dict name -> {"dtype","shape","data_offsets","data"} (o que read devolve)."""
    return {k: hashlib.sha256(v["data"]).hexdigest() for k, v in tensors.items()}


def to_write_list(tensors):
    """Converte o dict do read na lista de tuplas que o write espera, em ordem de offset."""
    order = sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"][0])
    return [(n, i["dtype"], list(i["shape"]), i["data"]) for n, i in order]


def ledger_row(path, tag):
    if not os.path.exists(path):
        return None
    hit = None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("tag") == tag:
            hit = r
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="diretorio do checkpoint")
    ap.add_argument("--st", default="", help="caminho do .safetensors (default: DIR/model.safetensors)")
    ap.add_argument("--ledger", default=os.path.join(REPO, "energy", "ledger.jsonl"))
    ap.add_argument("--tag", default="", help="tag do job no ledger de energia")
    ap.add_argument("--joules", type=float, default=None, help="J do job (se nao vier do ledger)")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=0)
    ap.add_argument("--eval", action="append", default=[], help="chave=valor (repetivel)")
    ap.add_argument("--lineage", action="append", default=[], help="chave=valor (repetivel)")
    ap.add_argument("--note", default="")
    ap.add_argument("--sidecar", action="store_true", default=True)
    a = ap.parse_args()

    rw = oracle()
    st = a.st or os.path.join(a.ckpt, "model.safetensors")
    if not os.path.exists(st):
        sys.exit(f"card_annotate: {st} nao existe (checkpoint npy nao tem card)")
    meta, tensors = rw.read(st)
    before = payload_hashes(tensors)

    new = dict(meta or {})
    card = {}
    try:
        card = json.loads(new.get("card", "{}"))
    except (ValueError, TypeError):
        card = {}
    card_energy = card.get("energy") or {}
    # AUTOMEDICAO: o train_run mede a propria energia (pacote fortran_energy) e
    # grava no card o delta EXATO entre o save anterior e este. Quando o card diz
    # energy.self_measured=1 essa medida tem precedencia sobre tudo que este script
    # faria (ledger ou --joules): o rateio linear por job e uma APROXIMACAO, e
    # sobrescrever o exato pelo aproximado seria andar para tras. Nesse caso NAO
    # sobrescrevemos NENHUM campo energy.* -- so preenchemos o que estiver faltando.
    self_measured = (str(new.get("energy.self_measured", "")) == "1" or
                     str(card_energy.get("self_measured", "")) == "1")
    energy = {}
    j = a.joules
    src = "cli"
    if self_measured:
        if a.joules is not None or a.tag:
            print("  aviso: card tem energy.self_measured=1 — ignorando --joules/--tag "
                  "(a medida do processo tem precedencia)", flush=True)
        filled = []

        def fill_missing(key, value):
            # Valores do __metadata__ sao SEMPRE string no safetensors (o writer
            # faz json_escape em cada um): numero tem que virar texto aqui, senao
            # o write morre no meio com 'float object is not iterable'.
            if f"energy.{key}" not in new:
                new[f"energy.{key}"] = value if isinstance(value, str) else json.dumps(value)
                filled.append(key)

        fill_missing("source", "self")
        fill_missing("attribution", "auto-medida pelo processo (delta exato desde o "
                                    "save anterior)")
        # J/Mtok do checkpoint, dos numeros do PROPRIO card (nunca do ledger)
        try:
            jj = float(new.get("energy.J", card_energy.get("J", "nan")))
            tok = float(card.get("tokens") or 0)
            if tok > 0 and jj == jj:
                fill_missing("J_per_Mtok", round(jj / (tok / 1e6), 1))
        except (ValueError, TypeError):
            pass
        n_kept = len([k for k in new if k.startswith("energy.")]) - len(filled)
        print(f"  energia AUTO-MEDIDA (energy.self_measured=1): {n_kept} campos "
              f"energy.* preservados, {len(filled)} preenchidos "
              f"({', '.join(filled) if filled else 'nenhum'}) — nada sobrescrito",
              flush=True)
    elif j is None and a.tag:
        row = ledger_row(a.ledger, a.tag)
        if row is None:
            print(f"  aviso: tag '{a.tag}' nao esta em {a.ledger} (energia fica vazia)",
                  flush=True)
        else:
            j = row.get("J")
            src = "ledger"
            energy["W_mean"] = row.get("W_mean")
            energy["wall_s"] = row.get("wall_s")
            energy["cpu_s"] = row.get("cpu_s")
            energy["job"] = a.tag
            if row.get("J_domains"):
                energy["J_domains"] = row["J_domains"]
    if not self_measured and j is not None:
        frac = 1.0
        if a.steps and a.total_steps:
            frac = a.steps / a.total_steps
            energy["attribution"] = (f"J_job x {a.steps}/{a.total_steps} "
                                     f"(rateio linear por passo)")
        else:
            energy["attribution"] = "J do job inteiro (sem rateio por passo)"
        energy["J_job"] = round(j, 1)
        energy["J"] = round(j * frac, 1)
        energy["source"] = src
        # J/token do checkpoint, usando os tokens que o propio card declara
        try:
            tok = float(card.get("tokens") or 0)
            if tok > 0:
                energy["J_per_Mtok"] = round(energy["J"] / (tok / 1e6), 1)
        except (ValueError, TypeError):
            pass
        for k, v in energy.items():
            new[f"energy.{k}"] = v if isinstance(v, str) else json.dumps(v)
    for item in a.eval:
        k, _, v = item.partition("=")
        new[f"eval.{k.strip()}"] = v.strip()
    for item in a.lineage:
        k, _, v = item.partition("=")
        new[f"lineage.{k.strip()}"] = v.strip()
    if a.note:
        new["annotate.note"] = a.note
    new["annotate.ts"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z")
    new["n_tensors"] = str(len(tensors))

    shutil.copy(st, st + ".bak")
    rw.write(st, to_write_list(tensors), metadata=new)
    meta2, tensors2 = rw.read(st)
    after = payload_hashes(tensors2)
    bad = [k for k in before if before[k] != after.get(k)]
    if bad or len(after) != len(before):
        shutil.move(st + ".bak", st)
        sys.exit(f"card_annotate: ABORTA — payloads mudaram ({bad[:3]}); arquivo restaurado")
    wsha = hashlib.sha256(open(st, "rb").read()).hexdigest()
    print(f"card_annotate: {os.path.basename(st)} — {len(tensors)} tensores intactos, "
          f"{len([k for k in new if k.startswith(('energy.', 'eval.', 'lineage.'))])} "
          f"campos de card")
    if "energy.J" in new:
        who = "auto-medida" if self_measured else f"externa ({new.get('energy.source', '?')})"
        print(f"  energia ({who}): J={new['energy.J']} "
              f"J_per_Mtok={new.get('energy.J_per_Mtok','-')} "
              f"({new.get('energy.attribution','')})")
    if a.sidecar:
        sc = os.path.join(a.ckpt, "card.json")
        json.dump({"weights": os.path.basename(st), "weights_sha256": wsha,
                   "metadata": new}, open(sc, "w"), indent=1, sort_keys=True)
        print(f"  sidecar: {sc} (com sha256 do arquivo de pesos: {wsha[:12]}…)")


if __name__ == "__main__":
    main()
