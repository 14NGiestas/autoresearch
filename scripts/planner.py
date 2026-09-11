#!/usr/bin/env python3
"""planner.py — calculadora de estimativas para ESCOLHER o modelo.

Três perguntas, com as constantes medidas nesta máquina:
  1. treina rápido?      dias = tokens x FLOPs_token / (soma dos boxes)
  2. cabe para inferir?  pesos (4 B/param fp32, 1 B int8) + cache KV
  3. tem dado para isso? Chinchilla pede ~20 tokens por parâmetro; com 4 épocas
                         do corpus, o teto de dados é corpus_tokens/5 params

Formulas (a primeira foi VALIDADA contra a realidade: previu 5,77 s/passo para
o modelo de 97,5M -- medido 5,77):
  N (params)          = 12*L*d^2 + 2*V*d          (12*d^2 por camada: 4 de atenção
                                                    + 8 de MLP; + embeddings/lm_head)
  FLOPs/token treino  = 6*N + 12*L*T*d            (atenção é O(T^2))
  FLOPs/token infer   = 2*N + 4*L*T*d
Constantes medidas: fermi 288 GFLOP/s (8 threads, fp32, nas formas do modelo),
halfbeast 187 GFLOP/s (10 threads). BPE comprime 2,50 bytes/token (medido).

Usage:
  scripts/planner.py                                   # cenário de hoje
  scripts/planner.py --data-tokens 500e6 --max-epochs 4 --hours 48
  scripts/planner.py --sizes 3,6,10,25,50,100 --T 1024
"""
import argparse
import math

V_DEFAULT = 8192          # vocabulário (BPE 8189 ranks + especiais)
FERMI = 288e9             # GFLOP/s medidos
HALFBEAST = 187e9
BYTES_PER_TOKEN = 2.502   # medido em prosa PT (P1)


def solve_d(N, L, V):
    """d tal que 12*L*d^2 + 2*V*d = N (a inversa de N(d))."""
    d = math.sqrt(N / (12.0 * L))
    for _ in range(60):
        d = math.sqrt(max(N - 2.0 * V * d, 1.0) / (12.0 * L))
    return d


def params_of(d, L, V=V_DEFAULT):
    return 12.0 * L * d * d + 2.0 * V * d


def flops_train(N, L, d, T):
    return 6.0 * N + 12.0 * L * T * d


def flops_infer(N, L, d, T):
    return 2.0 * N + 4.0 * L * T * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="3,6,10,25,50,100", help="M de params")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--data-tokens", type=float, default=53.7e6)
    ap.add_argument("--max-epochs", type=float, default=4.0)
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--ram-gb", type=float, default=15.0)
    ap.add_argument("--boxes", default="fermi,halfbeast")
    ap.add_argument("--bw-gbps", type=float, default=23.0,
                    help="banda efetiva de memória para inferência (medida: 23)")
    args = ap.parse_args()

    _validate()
    print()
    rate = 0.0
    for b in args.boxes.split(","):
        rate += {"fermi": FERMI, "halfbeast": HALFBEAST}.get(b.strip(), 0.0)
    sizes = [float(s) * 1e6 for s in args.sizes.split(",")]

    print(f"boxes: {args.boxes} = {rate/1e9:.0f} GFLOP/s (treino, compute-bound) | "
          f"inferência ~{args.bw_gbps:.0f} GB/s (bandwidth-bound) | T={args.T} | "
          f"dados: {args.data_tokens/1e6:.1f}M tokens | "
          f"limites: {args.max_epochs:.0f} épocas, {args.hours/24:.1f} dias, "
          f"{args.ram_gb:.0f} GB de RAM")
    print()
    hdr = (f"{'params':>8s} {'d':>4s} {'MFLOP/tok':>10s} {'Chinchilla':>11s} "
           f"{'épocas':>7s} {'dias':>6s} {'RAM fp32':>9s} {'RAM int8':>9s} "
           f"{'tok/s inf':>9s} {'veredito':>9s}")
    print(hdr)
    print("-" * len(hdr))
    best = None
    for N in sizes:
        d = solve_d(N, args.layers, V_DEFAULT)
        ft = flops_train(N, args.layers, d, args.T)
        need = 20.0 * N
        ep = need / args.data_tokens
        days = need * ft / rate / 86400.0
        ram32 = 4.0 * N / 1e9
        kv = 2.0 * args.layers * args.T * d * 4 / 1e9      # 1 sequência
        ram8 = 1.0 * N / 1e9
        # INFERÊNCIA de uma sequência é limitada por BANDA, não por FLOPs: cada
        # token exige ler todos os pesos. Calibrado no repl (97,5M fp32 = 390 MB
        # por token, medidos 60 tok/s em 4 threads => ~23 GB/s de banda efetiva).
        tps = args.bw_gbps * 1e9 / (4.0 * N)
        ok = []
        if ep <= args.max_epochs:
            ok.append("dado")
        if days <= args.hours / 24.0:
            ok.append("tempo")
        if ram32 + kv <= args.ram_gb:
            ok.append("RAM")
        if len(ok) == 3:
            v = "OK"
        else:
            fail = [n for n, o in (("dado", ep <= args.max_epochs),
                                   ("tempo", days <= args.hours / 24.0),
                                   ("RAM", ram32 + kv <= args.ram_gb)) if not o]
            v = "x " + "+".join(fail)
        print(f"{N/1e6:7.1f}M {d:4.0f} {ft/1e6:10.0f} {need/1e9:10.2f}B "
              f"{ep:7.1f} {days:6.2f} {ram32:8.2f}G {ram8:8.2f}G "
              f"{tps:8.0f} {v:>9s}")
        if len(ok) == 3 and (best is None or N > best[0]):
            best = (N, d, days, ep)

    print()
    if best:
        N, d, days, ep = best
        print(f"VEREDITO: {N/1e6:.0f}M params (d={d:.0f}, L={args.layers}, "
              f"T={args.T}) -- {days:.2f} dias, {ep:.1f} épocas. "
              f"É o maior que cabe em tempo + dados + RAM.")
    else:
        print("VEREDITO: nenhum tamanho fecha os três limites -- relaxe tempo, "
              "RAM, épocas ou baixe mais dados.")

    # ---- dados: quanto texto, e o que mais dados comprariam -----------------
    print()
    print("DADOS (o recurso escasso):")
    print(f"  hoje: {args.data_tokens/1e6:.1f}M tokens = "
          f"{args.data_tokens*BYTES_PER_TOKEN/1e6:.0f}M bytes = "
          f"{args.data_tokens*BYTES_PER_TOKEN/1e9/0.0004:.0f} livros de ~400 KB")
    print(f"  em 1 época, 20 tok/param: sustenta {args.data_tokens/20/1e6:.1f}M params")
    print(f"  em {args.max_epochs:.0f} épocas:          sustenta "
          f"{args.data_tokens*args.max_epochs/20/1e6:.1f}M params")
    for target in (25e6, 50e6, 100e6):
        need = 20.0 * target
        d = solve_d(target, args.layers, V_DEFAULT)
        days = need * flops_train(target, args.layers, d, args.T) / rate / 86400.0
        gb = need * BYTES_PER_TOKEN / 1e9
        print(f"  para {target/1e6:5.0f}M params em 1 época: {need/1e9:.2f}B tokens "
              f"= {gb:.1f} GB de texto (~{gb/0.0004:,.0f} livros) -> {days:.0f} dias "
              f"nos dois boxes")


def _validate():
    """A calculadora tem de reproduzir o que já medimos, senão não serve."""
    N, L, T = 97.5e6, 12, 2048
    d = 768.0
    ft = flops_train(N, L, d, T)
    print(f"  validação treino: {ft/1e6:.0f} MFLOP/token x 2049 tokens / "
          f"{(FERMI)/1e9:.0f} GFLOP/s = {ft*2049/FERMI:.2f} s/passo "
          f"(medido: 5,77 s/passo)")
    print(f"  validação infer.: 4N = {4*N/1e6:.0f} MB/token / 23 GB/s = "
          f"{23e9/(4*N):.0f} tok/s (medido no repl: 60 tok/s em 4 threads)")


if __name__ == "__main__":
    main()
