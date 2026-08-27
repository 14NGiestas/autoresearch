"""HEP — Hypothesis Evolution Protocol registry.

Implements the auditable registry from arXiv:2607.09195v1 (Toward Auditable
AI Scientists). Every event is appended to an append-only, hash-chained log
(like the paper's event-sourced HEP Registry), so each hypothesis carries an
unbroken, verifiable record from proposal to verdict.

Usage (CLI):
  python3 hep.py propose  --statement "..." --prior 0.5 --mechanism de-novo
  python3 hep.py evidence --hyp hyp_XXXX --kind simulation --direction supports --bpb 1.23 --prior 0.5 --updated 0.55 --rationale "..."
  python3 hep.py transition --hyp hyp_XXXX --state supported
  python3 hep.py status

Or import: from hep import HEP; h = HEP(); hid = h.propose(...).
"""

import argparse
import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone

REGISTRY = os.path.join(os.path.dirname(__file__), "hep", "registry.jsonl")

VALID_STATES = {"proposed", "under_test", "supported", "refuted", "dormant"}
VALID_MECH = {"de-novo", "inspired-by", "refine", "merge"}
VALID_DIR = {"supports", "refutes", "inconclusive"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hid():
    return "hyp_" + secrets.token_hex(3)


class HEP:
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
        event = {
            "seq": self._seq,
            "prev": self._prev,
            "type": etype,
            "ts": _now(),
            "payload": payload,
        }
        h = hashlib.sha256(
            (self._prev + json.dumps(payload, sort_keys=True)).encode()
        ).hexdigest()
        event["hash"] = h
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")
        self._prev = h
        return event

    def propose(self, statement, prior, mechanism="de-novo", parents=None,
                testable=None):
        if mechanism not in VALID_MECH:
            raise ValueError(f"mechanism must be one of {VALID_MECH}")
        hid = _hid()
        self._append("propose", {
            "hyp": hid,
            "statement": statement,
            "prior": float(prior),
            "mechanism": mechanism,
            "parents": parents or [],
            "testable_observable": testable,
            "state": "proposed",
        })
        return hid

    def evidence(self, hyp, kind, direction, prior, updated, rationale,
                 source="", bpb=None, commit=""):
        if direction not in VALID_DIR:
            raise ValueError(f"direction must be one of {VALID_DIR}")
        return self._append("evidence", {
            "hyp": hyp,
            "kind": kind,
            "direction": direction,
            "prior": float(prior),
            "updated": float(updated),
            "bpb": bpb,
            "commit": commit,
            "rationale": rationale,
            "source": source,
        })

    def transition(self, hyp, state):
        if state not in VALID_STATES:
            raise ValueError(f"state must be one of {VALID_STATES}")
        return self._append("transition", {"hyp": hyp, "state": state})

    def status(self):
        hyps = {}
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                p = e["payload"]
                if e["type"] == "propose":
                    hyps[p["hyp"]] = {
                        "statement": p["statement"],
                        "prior": p["prior"],
                        "mechanism": p["mechanism"],
                        "parents": p["parents"],
                        "state": p["state"],
                        "belief": p["prior"],
                        "evidence": [],
                    }
                elif e["type"] == "evidence":
                    if p["hyp"] in hyps:
                        hyps[p["hyp"]]["belief"] = p["updated"]
                        hyps[p["hyp"]]["evidence"].append(p)
                elif e["type"] == "transition":
                    if p["hyp"] in hyps:
                        hyps[p["hyp"]]["state"] = p["state"]
        return hyps


def _print_status(h):
    hyps = h.status()
    if not hyps:
        print("(no hypotheses yet)")
        return
    print(f"{'hyp':<12}{'state':<12}{'belief':>7}  mechanism   statement")
    for hid, d in hyps.items():
        print(f"{hid:<12}{d['state']:<12}{d['belief']:>7.2f}  {d['mechanism']:<10} {d['statement'][:60]}")
    print(f"\n{len(hyps)} hypotheses")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("propose")
    p.add_argument("--statement", required=True)
    p.add_argument("--prior", type=float, required=True)
    p.add_argument("--mechanism", default="de-novo")
    p.add_argument("--parents", default="")
    p.add_argument("--testable", default="")

    e = sub.add_parser("evidence")
    e.add_argument("--hyp", required=True)
    e.add_argument("--kind", default="analysis")
    e.add_argument("--direction", required=True)
    e.add_argument("--prior", type=float, required=True)
    e.add_argument("--updated", type=float, required=True)
    e.add_argument("--bpb", type=float, default=None)
    e.add_argument("--commit", default="")
    e.add_argument("--rationale", required=True)
    e.add_argument("--source", default="run.log")

    t = sub.add_parser("transition")
    t.add_argument("--hyp", required=True)
    t.add_argument("--state", required=True)

    sub.add_parser("status")

    args = ap.parse_args()
    h = HEP()
    if args.cmd == "propose":
        parents = [x for x in args.parents.split(",") if x]
        hid = h.propose(args.statement, args.prior, args.mechanism, parents, args.testable)
        print("proposed", hid)
    elif args.cmd == "evidence":
        h.evidence(args.hyp, args.kind, args.direction, args.prior,
                   args.updated, args.rationale, args.source, args.bpb, args.commit)
        print("evidence recorded for", args.hyp)
    elif args.cmd == "transition":
        h.transition(args.hyp, args.state)
        print("transitioned", args.hyp, "->", args.state)
    elif args.cmd == "status":
        _print_status(h)


if __name__ == "__main__":
    main()
