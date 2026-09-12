#!/usr/bin/env python3
"""ckpt.py — model registry: genealogia auditável de checkpoints, como o HEP.

Checkpoints moram fora do git (grandes); o repo guarda SÓ este registro
(ckpt/registry.jsonl), com hash-chain idêntico ao do hep.py:
  hash = sha256(prev + json(payload, sort_keys=True))

Cada entrada "register" identifica um checkpoint pelo CONTEÚDO
(id = ckpt_<12 hex do manifesto sha256>, determinístico e verificável),
mais a linhagem (parents), a fatia de corpus, o código (git sha) e a métrica.

Comandos:
  ckpt.py register --dir DIR --label L --parents ID,.. --machine M \\
      --rows-file F --start-row N --ntrain N --steps N --lr X \\
      --bpb X --eval-spec S --note S [--code-sha SHA]
  ckpt.py show [ID]            # pretty-print (tudo, ou um checkpoint)
  ckpt.py lineage ID           # cadeia de pais até a raiz
  ckpt.py verify ID            # re-hash dos arquivos, compara com o registro

Campos desconhecidos ficam null COM motivo em --note. Honestidade > completude:
um pai "provável" sem fonte é pior que parent ausente documentado.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REGISTRY = os.path.join(os.path.dirname(__file__), "ckpt", "registry.jsonl")


def _now():
    return datetime.now(timezone.utc).isoformat()


class Registry:
    def __init__(self, path=REGISTRY):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._seq, self._prev = self._tail()

    def _tail(self):
        seq, prev = 0, "0" * 64
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    seq = e["seq"]
                    prev = e["hash"]
        return seq, prev

    def _append(self, etype, payload):
        self._seq += 1
        event = {"seq": self._seq, "prev": self._prev, "type": etype,
                 "ts": _now(), "payload": payload}
        h = hashlib.sha256(
            (self._prev + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
        event["hash"] = h
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")
        self._prev = h
        return event

    def _sha256_file(self, path, n=1 << 20):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(n)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    def register(self, d, label, parents, machine, path, origin, rows_file,
                 start_row, ntrain, steps, lr, bpb, eval_spec, note, code_sha,
                 backfill_arch=False):
        files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
        if not files:
            raise SystemExit(f"sem .npy em {d}")
        manifest, total = [], 0
        for f in files:
            p = os.path.join(d, f)
            sz = os.path.getsize(p)
            total += sz
            manifest.append({"file": f, "bytes": sz,
                             "sha256": self._sha256_file(p)})
        manifest_hash = hashlib.sha256(
            json.dumps([m["sha256"] + "  " + m["file"] for m in manifest],
                       sort_keys=True).encode()).hexdigest()
        ckpt_id = "ckpt_" + manifest_hash[:12]
        arch = None
        ap = os.path.join(d, "arch.txt")
        if os.path.exists(ap):
            arch = open(ap).read().strip()
        elif backfill_arch:
            arch = None  # registrado sem arch.txt; formas vão em note
        payload = {
            "ckpt": ckpt_id, "label": label, "parents": parents,
            "machine": machine, "path": path, "origin": origin,
            "code": {"git_sha": code_sha},
            "arch": {"arch_txt": arch},
            "corpus": {"rows_file": rows_file, "start_row": start_row,
                       "ntrain": ntrain, "steps": steps, "lr": lr},
            "files": {"n": len(files), "bytes_total": total,
                      "manifest_sha256": manifest_hash, "files": manifest},
            "eval": {"bpb": bpb, "spec": eval_spec},
            "note": note,
        }
        ev = self._append("register", payload)
        print(f"{ckpt_id}  {label}  ({len(files)} files, "
              f"{total/1e6:.0f} MB, manifest {manifest_hash[:12]})")
        return ev

    def _all(self):
        out = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    if e["type"] == "register":
                        out[e["payload"]["ckpt"]] = e["payload"]
        return out

    def show(self, cid=None):
        allc = self._all()
        ids = [cid] if cid else list(allc)
        for i in ids:
            p = allc.get(i)
            if not p:
                print(f"desconhecido: {i}")
                continue
            e = p.get("eval", {}) or {}
            c = p.get("corpus", {}) or {}
            print(f"{p['ckpt']}  {p.get('label')}")
            print(f"  pais: {p.get('parents') or '(raiz do registro)'}")
            print(f"  maquina: {p.get('machine')}  path: {p.get('path')}")
            print(f"  codigo: {(p.get('code') or {}).get('git_sha')}")
            print(f"  corpus: {c.get('rows_file')} start={c.get('start_row')} "
                  f"ntrain={c.get('ntrain')} steps={c.get('steps')} lr={c.get('lr')}")
            print(f"  arquivos: {(p.get('files') or {}).get('n')} "
                  f"({(p.get('files') or {}).get('bytes_total', 0)/1e6:.0f} MB)")
            print(f"  bpb: {e.get('bpb')}  [{e.get('spec')}]")
            if p.get("note"):
                print(f"  nota: {p['note']}")

    def lineage(self, cid):
        allc = self._all()
        depth = 0
        seen = set()
        while cid and cid not in seen:
            seen.add(cid)
            p = allc.get(cid)
            if not p:
                print("  " * depth + f"{cid}  (fora do registro)")
                break
            e = p.get("eval", {}) or {}
            print("  " * depth + f"{p['ckpt']}  {p.get('label')}  "
                  f"bpb={e.get('bpb')}")
            ps = p.get("parents") or []
            if len(ps) != 1:
                for q in ps[1:]:
                    print("  " * (depth + 1) + f"+ ramo: {q}")
            cid = ps[0] if ps else None
            depth += 1

    def tree(self):
        allc = self._all()
        if not allc:
            print("(registro vazio)")
            return
        children = {}
        for cid, p in allc.items():
            for par in p.get("parents") or []:
                children.setdefault(par, []).append(cid)
        for v in children.values():
            v.sort()
        roots = [cid for cid, p in allc.items()
                 if not p.get("parents") or
                 not any(q in allc for q in p["parents"])]
        try:
            best = min((c for c in allc.values()
                          if (c.get("eval") or {}).get("bpb") is not None),
                         key=lambda p: float(p["eval"]["bpb"]))["ckpt"]
        except ValueError:
            best = None

        def label(cid):
            p = allc[cid]
            e = p.get("eval") or {}
            try:
                bs = f"{float(e['bpb']):.5f}"
            except (TypeError, ValueError, KeyError):
                bs = "?"
            star = " ★" if cid == best else ""
            return (f"{p.get('label')}{star}  [bpb={bs}]  "
                    f"({p.get('machine')})  {cid}")

        visited = set()

        def walk(cid, prefix):
            kids = [k for k in children.get(cid, []) if k in allc]
            ext = [k for k in children.get(cid, []) if k not in allc]
            for n, k in enumerate(kids + ext):
                last = (n == len(kids + ext) - 1)
                elbow = "└── " if last else "├── "
                if k not in allc:
                    print(prefix + elbow + f"{k}  (fora do registro)")
                    continue
                if k in visited:
                    print(prefix + elbow + f"↩ {k} (ver acima)")
                    continue
                visited.add(k)
                print(prefix + elbow + label(k))
                walk(k, prefix + ("    " if last else "│   "))

        for r in sorted(roots):
            visited.add(r)
            print(label(r))
            walk(r, "")

    def verify(self, cid):
        allc = self._all()
        p = allc.get(cid)
        if not p:
            raise SystemExit(f"desconhecido: {cid}")
        d = p["path"]
        ok, bad = True, []
        for m in (p.get("files") or {}).get("files", []):
            fp = os.path.join(d, m["file"])
            if not os.path.exists(fp):
                ok = False
                bad.append(m["file"] + " AUSENTE")
                continue
            if self._sha256_file(fp) != m["sha256"]:
                ok = False
                bad.append(m["file"] + " ALTERADO")
        print(("OK  " if ok else "FALHA ") + f"{cid} ({p.get('label')})")
        for b in bad[:10]:
            print(f"  {b}")
        return ok


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--dir", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--parents", default="")
    r.add_argument("--machine", default="")
    r.add_argument("--path", default="")
    r.add_argument("--origin", default="")
    r.add_argument("--rows-file", default=None)
    r.add_argument("--start-row", type=int, default=None)
    r.add_argument("--ntrain", type=int, default=None)
    r.add_argument("--steps", type=int, default=None)
    r.add_argument("--lr", type=float, default=None)
    r.add_argument("--bpb", type=float, default=None)
    r.add_argument("--eval-spec", default=None)
    r.add_argument("--note", default="")
    r.add_argument("--code-sha", default=None)
    sub.add_parser("show").add_argument("id", nargs="?")
    l = sub.add_parser("lineage")
    l.add_argument("id")
    sub.add_parser("tree")
    v = sub.add_parser("verify")
    v.add_argument("id")
    a = ap.parse_args()
    reg = Registry()
    if a.cmd == "register":
        if a.code_sha is None:
            try:
                a.code_sha = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True).stdout.strip() or None
            except Exception:
                a.code_sha = None
        reg.register(a.dir, a.label,
                     [p for p in a.parents.split(",") if p] or [],
                     a.machine or None, a.path or a.dir, a.origin or None,
                     a.rows_file, a.start_row, a.ntrain, a.steps, a.lr,
                     a.bpb, a.eval_spec, a.note, a.code_sha)
    elif a.cmd == "show":
        reg.show(a.id)
    elif a.cmd == "lineage":
        reg.lineage(a.id)
    elif a.cmd == "tree":
        reg.tree()
    elif a.cmd == "verify":
        sys.exit(0 if reg.verify(a.id) else 1)


if __name__ == "__main__":
    main()
