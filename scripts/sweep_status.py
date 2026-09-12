#!/usr/bin/env python3
"""sweep_status.py — observabilidade da varredura de dose-resposta (e vizinhança).

Por que existe: os logs brutos existem, mas ninguém agrega. Hoje três falhas
custaram tempo e NENHUMA aparecia no log: um build_mix que escrevia só-wiki em
silêncio, um binário de outra arquitetura, e um pkill que casava com a própria
linha de comando. Aqui a informação que decide fica visível e COMPARÁVEL:

  - por braço: estado no Slurm, passos/total, ritmo medido, ETA, último val bpb
  - a CURVA em formação (val bpb por fração de wiki), que é o resultado do estudo
  - checagem de SANIDADE dos insumos: contagem de linhas por braço (o portão que
    pegou o corpus só-wiki) e existência do init/arch.txt

Uso: scripts/sweep_status.py [--jobs 5] [--watch N]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRACS = [0, 25, 50, 75, 100]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def slurm_state(jobid):
    out = sh(f"squeue -h -j {jobid} -o '%T %M' 2>/dev/null")
    if not out:
        return "FIM", "-"
    p = out.split()
    return p[0], (p[1] if len(p) > 1 else "-")


def hms(sec):
    sec = int(sec)
    return f"{sec//3600}h{(sec%3600)//60:02d}m" if sec >= 3600 else f"{sec//60}m{sec%60:02d}s"


def parse_log(path):
    """Conta SÓ a última execução do arquivo.

    Um arquivo de log pode conter várias submissões (mesmo nome por %a). Foi assim
    que um ETA de 53h apareceu: passos de uma corrida cancelada somados ao tempo de
    outra. Cada execução imprime um cabeçalho 'arm wiki=', então cortamos nele.
    """
    steps = vals = 0
    last_val = None
    if os.path.exists(path):
        lines = open(path, errors="ignore").read().splitlines()
        cut = 0
        for idx, ln in enumerate(lines):
            if ln.startswith("arm wiki="):
                cut = idx
        for line in lines[cut:]:
            if line.startswith("step "):
                steps += 1
            elif line.startswith("val @"):
                p = line.split()
                vals += 1
                last_val = (p[2], p[3])
    return steps, vals, last_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--watch", type=int, default=0, help="repete a cada N s")
    args = ap.parse_args()
    while True:
        total_steps = 7812
        print(f"\n=== varredura de dose-resposta  3M params  T=1024  {time.strftime('%H:%M:%S')}")
        print(f"  {'braço':>5s} {'wik%':>5s} {'estado':>9s} {'passos':>13s} "
              f"{'ritmo':>8s} {'ETA':>7s} {'val bpb':>9s} {'probes':>6s}")
        curve = []
        for i, f in enumerate(FRACS):
            # o log mais recente daquele braço (nome pode ter JOBID após este fix)
            cands = sorted(glob.glob(f"{ROOT}/logs/sweep3m_*_{i}.log") +
                           glob.glob(f"{ROOT}/logs/sweep3m_{i}.log"),
                           key=os.path.getmtime)
            log = cands[-1] if cands else f"{ROOT}/logs/sweep3m_{i}.log"
            steps, vals, last_val = parse_log(log)
            state, used = slurm_state(f"{args.jobs}_{i}")
            # Ritmo medido pelo NASCIMENTO do log (getctime), nao pelo tempo do
            # squeue: ler %M deu 28 s/passo num job que andava a 0,42 s/passo, e o
            # ETA absurdo (22h) quase me fez acreditar nele. O log nasce quando o
            # job comeca, entao (agora - ctime)/passos e' o ritmo de verdade.
            rate = eta = None
            if steps and os.path.exists(log) and state == "RUNNING":
                sec = time.time() - os.path.getctime(log)
                if sec > 5:
                    rate = sec / steps
                    eta = rate * (total_steps - steps)
            print(f"  {args.jobs}_{i:<3d} {f:>4d}% {state:>9s} "
                  f"{steps:>6d}/{total_steps:<6d} "
                  f"{(f'{rate:.2f}s' if rate else '-'):>8s} "
                  f"{(hms(eta) if eta else '-'):>7s} "
                  f"{('%.5f' % float(last_val[1]) if last_val else '-'):>9s} "
                  f"{vals:>6d}")
            if last_val:
                curve.append((f, float(last_val[1])))
        if curve:
            best = min(curve, key=lambda x: x[1])
            print("  curva em formação (val bpb em prosa literária held-out):")
            print("    " + "   ".join(f"{f}%: {v:.4f}" for f, v in curve))
            print(f"    menor até agora: {best[0]}% de wiki ({best[1]:.4f})")
        # sanidade dos insumos: o portão que pegou o corpus só-wiki
        bad = [f for f in FRACS
               if os.path.exists(f"/tmp/mix/train_f{f}.txt")
               and sum(1 for _ in open(f"/tmp/mix/train_f{f}.txt")) != total_steps]
        if bad:
            print(f"  ATENÇÃO: braços com contagem de linhas errada: {bad}"
                  f" (esperado {total_steps})")
        if not os.path.exists("/tmp/mix/init3m/arch.txt"):
            print("  ATENÇÃO: init sem arch.txt (require_arch vai abortar)")
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
