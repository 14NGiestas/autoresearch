#!/usr/bin/env python3
"""energy_run.py — contabilidade de energia dos treinos (J por token).

Por que existe: bpb mede qualidade, J/token mede PRECO. Sem o segundo eixo,
qualquer conversa de escala (1B em 6 meses, compor vs treinar direto) fica no
chute. Este wrapper mede o pacote (APU/CPU) por amostragem e atribui a energia
ao PROCESSO (wall + cpu-seconds do filho), nao a maquina inteira.

Sensores, tentados em ordem:
  1. hwmon com nome amdgpu/zenpower/amd_energy/rapl -> power1_input (uW)
     [fermi: amdgpu no APU 8745HS reporta o PPT do pacote, ~40 W sob carga]
  2. intel-rapl energy_uj por delta (hoje root-only em fermi e halfbeast)
  3. nenhum -> mode=cpu_only: registra cpu_s; a energia sai de cpu_s * W_ref

Registros vao para energy/ledger.jsonl (uma linha por run).

Uso:
  scripts/energy_run.py idle --seconds 60
  scripts/energy_run.py run --tag compose/disj2_s0 --tokens 4000000 -- CMD ...
  scripts/energy_run.py report [--since 2026-09-15]
  scripts/energy_run.py estimate --tag f0 --cpu-s 12400 --watts 45   # retro
"""
import argparse
import glob
import json
import os
import resource
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(REPO, "energy", "ledger.jsonl")
PREF = ("amdgpu", "zenpower", "amd_energy", "k10temp", "coretemp", "rapl")


def find_sensor():
    """(kind, path) do melhor sensor disponivel, ou (None, None)."""
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            name = open(os.path.join(h, "name")).read().strip().lower()
        except OSError:
            continue
        if any(p in name for p in PREF):
            for f in ("power1_input", "power1_average"):
                p = os.path.join(h, f)
                if os.access(p, os.R_OK):
                    return "hwmon", p
    for h in sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")) + \
            sorted(glob.glob("/sys/class/powercap/amd_energy/*/energy_uj")):
        if os.access(h, os.R_OK):
            return "rapl", h
    return None, None


def read_w(path):
    try:
        return float(open(path).read().strip()) / 1e6      # uW -> W
    except (OSError, ValueError):
        return None


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def append(rec):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def base(tag, mode, note=""):
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag,
            "host": os.uname().nodename, "mode": mode, "note": note}


def cmd_idle(a):
    kind, path = find_sensor()
    if not kind:
        sys.exit("nenhum sensor de potencia disponivel nesta maquina")
    ws = []
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        w = read_w(path)
        if w is not None:
            ws.append(w)
        time.sleep(1.0)
    rec = base("idle", kind, f"sensor={path}")
    rec.update({"wall_s": round(time.time() - t0, 2), "n": len(ws),
                "W_mean": round(sum(ws) / len(ws), 3),
                "W_min": round(min(ws), 3), "W_max": round(max(ws), 3)})
    append(rec)
    print(json.dumps(rec, indent=1))


def cmd_run(a):
    kind, path = find_sensor()
    e0 = None
    if kind == "rapl":
        e0 = float(open(path).read().strip())
    ws, ts = [], []
    t0 = time.time()
    proc = subprocess.Popen(a.cmd)
    last = t0
    while proc.poll() is None:
        if kind == "hwmon":
            w = read_w(path)
            if w is not None:
                ws.append(w)
                ts.append(time.time() - t0)
        time.sleep(a.interval)
    rc = proc.returncode
    t1 = time.time()
    wall = t1 - t0
    cpus = cpu_seconds()
    rec = base(a.tag, kind or "cpu_only", a.note)
    rec.update({"rc": rc, "wall_s": round(wall, 2),
                "cpu_s": round(cpus, 2), "cmd": " ".join(a.cmd),
                "cmd_id": a.cmd_id})
    if kind == "rapl":
        e1 = float(open(path).read().strip())
        span = 65532610987.0 if "intel" in path else 262143328850.0
        d = (e1 - e0) % span
        rec.update({"sensor": path, "J": round(d / 1e6, 1)})
    if ws:
        # trapezoidal em (t, W)
        j = 0.0
        for i in range(1, len(ws)):
            j += 0.5 * (ws[i] + ws[i - 1]) * (ts[i] - ts[i - 1])
        j += ws[-1] * max(0.0, wall - ts[-1]) + ws[0] * max(0.0, ts[0])
        rec.update({"sensor": path, "J": round(j, 1), "n": len(ws),
                    "W_mean": round(sum(ws) / len(ws), 3),
                    "W_max": round(max(ws), 3)})
    if a.idle_w and "J" in rec:
        rec["J_net"] = round(max(0.0, rec["J"] - a.idle_w * wall), 1)
    if a.tokens:
        rec["tokens"] = a.tokens
        if "J" in rec:
            rec["J_per_Mtok"] = round(rec["J"] / (a.tokens / 1e6), 2)
        rec["cpu_s_per_Mtok"] = round(cpus / (a.tokens / 1e6), 2)
    if "J_net" in rec and a.tokens:
        rec["J_net_per_Mtok"] = round(rec["J_net"] / (a.tokens / 1e6), 2)
    append(rec)
    print("energy: " + json.dumps({k: v for k, v in rec.items()
                                   if k not in ("cmd",)}), flush=True)
    return rc


def cmd_report(a):
    rows = []
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if a.since and r["ts"] < a.since:
                continue
            rows.append(r)
    tot_j = tot_cpu = 0.0
    print(f"{'ts':16s} {'tag':22s} {'mode':9s} {'wall_s':>8s} {'cpu_s':>9s} "
          f"{'W':>6s} {'J':>9s} {'J/Mtok':>8s}")
    for r in rows:
        j = r.get("J", 0.0) or 0.0
        tot_j += j
        tot_cpu += r.get("cpu_s", 0.0) or 0.0
        print(f"{r['ts'][:16]:16s} {r.get('tag','')[:22]:22s} "
              f"{r.get('mode',''):9s} {r.get('wall_s',0):8.1f} "
              f"{r.get('cpu_s',0):9.1f} {r.get('W_mean',0):6.2f} {j:9.0f} "
              f"{r.get('J_per_Mtok',0):8.2f}")
    print(f"\ntotal: {tot_j/3.6e6:.3f} kWh medidos, {tot_cpu/3600:.2f} core-h "
          f"({len(rows)} registros)")


def cmd_estimate(a):
    j = a.cpu_s * a.watts
    rec = base(a.tag, "estimate", a.note or "retroativo: cpu_s x W_ref")
    rec.update({"cpu_s": a.cpu_s, "W_ref": a.watts, "J": round(j, 1),
                "J_per_Mtok": round(j / (a.tokens / 1e6), 2) if a.tokens else None,
                "tokens": a.tokens})
    append(rec)
    print(json.dumps(rec))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tag", required=True)
    p.add_argument("--tokens", type=float, default=0)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--idle-w", type=float, default=0.0)
    p.add_argument("--note", default="")
    p.add_argument("--cmd-id", default="")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("idle")
    p.add_argument("--seconds", type=float, default=60)
    p.set_defaults(fn=cmd_idle)

    p = sub.add_parser("report")
    p.add_argument("--since", default="")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("estimate")
    p.add_argument("--tag", required=True)
    p.add_argument("--cpu-s", type=float, required=True)
    p.add_argument("--watts", type=float, required=True)
    p.add_argument("--tokens", type=float, default=0)
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_estimate)

    a = ap.parse_args()
    if a.what == "run":
        if a.cmd and a.cmd[0] == "--":
            a.cmd = a.cmd[1:]
        if not a.cmd:
            sys.exit("faltou o comando (use -- CMD ...)")
        sys.exit(cmd_run(a))
    a.fn(a)


if __name__ == "__main__":
    main()
