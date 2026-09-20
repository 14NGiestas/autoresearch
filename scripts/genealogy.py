# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
import json
import os
from datetime import datetime

REGISTRY = "hep/registry.jsonl"

def load_registry():
    events = []
    if not os.path.exists(REGISTRY):
        return events
    with open(REGISTRY, "r") as f:
        for line in f:
            events.append(json.loads(line))
    return events

def print_tree():
    events = load_registry()
    hyps = {}
    
    # Primeiro pass: indexar as hipóteses
    for ev in events:
        if ev["type"] == "propose":
            h = ev["payload"]
            hid = h["hyp"]
            hyps[hid] = {
                "statement": h["statement"],
                "mechanism": h["mechanism"],
                "parents": h.get("parents", []),
                "evidence": [],
                "state": "proposed"
            }
        elif ev["type"] == "evidence":
            hid = ev["payload"]["hyp"]
            if hid in hyps:
                hyps[hid]["evidence"].append(ev["payload"])
        elif ev["type"] == "transition":
            hid = ev["payload"]["hyp"]
            if hid in hyps:
                hyps[hid]["state"] = ev["payload"]["state"]

    print("\n=== GENEALOGIA DE TREINAMENTO (HEP) ===\n")
    
    # Função recursiva simples para mostrar a árvore
    def show(hid, indent="", visited=None):
        if visited is None: visited = set()
        if hid in visited: return
        visited.add(hid)
        
        h = hyps.get(hid)
        if not h: return
        
        state_icon = {
            "supported": "✅",
            "refuted": "❌",
            "under_test": "🧪",
            "dormant": "💤",
            "proposed": "📝"
        }.get(h["state"], "❓")
        
        # Pega o último bpb medido
        bpb = "N/A"
        if h["evidence"]:
            last_ev = h["evidence"][-1]
            if last_ev.get("bpb"):
                bpb = f"{last_ev['bpb']:.5f}"

        print(f"{indent}{state_icon} {hid} [{h['mechanism']}] bpb={bpb}")
        print(f"{indent}   \"{h['statement'][:80]}...\"")
        
        # Procura filhos (quem tem hid como pai)
        children = [cid for cid, ch in hyps.items() if hid in ch["parents"]]
        for cid in children:
            show(cid, indent + "    │", visited)

    # Começa das raízes (sem pais ou pais não encontrados)
    roots = [hid for hid, h in hyps.items() if not h["parents"]]
    for rid in roots:
        show(rid)

if __name__ == "__main__":
    print_tree()
