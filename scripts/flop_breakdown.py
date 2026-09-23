#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Onde os FLOPs do nosso passo vao -- estatico, das formas da arquitetura.

Por que isto importa: o estimador (speedup_estimate.py) preca tudo com 6*P
FLOPs por token, a convencao classica de fwd+bwd que conta SO' as camadas
lineares. Em modelos grandes (1B, contexto 8k) isso e' uma boa aproximacao,
porque os lineares dominam. No NOSSO regime, nao: contexto 1024 com d_model 96
faz a atencao e a cabeca de vocabulario pesarem muito.

Este script conta, por componente, os MACs por token (forward), e aplica o
fator 3 do fwd+bwd. Nada e' medido: e' aritmetica das formas.

Uso: uv run scripts/flop_breakdown.py
"""

D = 96          # d_model
NH = 6          # heads
NKV = 2         # kv heads (GQA)
HD = D // NH    # 16
NL = 12         # layers
T = 1024        # contexto
V = 8192        # vocab
DFF = 4 * D     # feed-forward interno (384)
B = 1

# MACs por token (forward), por componente, por camada quando indicado
comp = {}
# atencao: score QK^T e saida PV. Por cabeca de query, T pares, cada par 2*HD
comp["atencao (QK^T + PV)"] = NL * NH * T * 2 * HD
# lineares de atencao: q, k, v, proj
comp["lineares de atencao (q,k,v,o)"] = NL * D * D * (NH * HD / D + 2 * NKV * HD / D + 1)
# MLP: up e down
comp["MLP (up + down)"] = NL * 2 * D * DFF
# cabeca de vocabulario (logits)
comp["cabeca de vocab (logits)"] = V * D
# norms (rmsnorm: ~1 passada elementwise, contada como MAC por elemento)
comp["rmsnorm + residual"] = NL * 2 * D + D

total_mac = sum(comp.values())
print("=" * 74)
print("O NOSSO PASSO, POR COMPONENTE  (d96, T=1024, V=8192, L=12)")
print("=" * 74)
print(f"  {'componente':<30} {'MACs/token':>14} {'%':>7} {'MFLOP/token':>12}")
for k in sorted(comp, key=lambda x: -comp[x]):
    m = comp[k]
    print(f"  {k:<30} {m:>14,.0f} {100*m/total_mac:>6.1f}% {3*2*m/1e6:>12.2f}")
print(f"  {'TOTAL (fwd+bwd, x3)':<30} {total_mac:>14,.0f} {'100.0%':>7} "
      f"{3*2*total_mac/1e6:>12.2f}")
print()
six_p = 6 * 2.75e6
print(f"  a convencao 6*P (so' lineares, o que o estimador usa): {six_p/1e6:.2f} MFLOP/token")
print(f"  o nosso passo de verdade (esta conta):                {3*2*total_mac/1e6:.2f} MFLOP/token")
print(f"  => 6*P subestima em {(3*2*total_mac)/six_p:.2f}x")
print()
lin = comp["lineares de atencao (q,k,v,o)"] + comp["MLP (up + down)"]
print(f"  lineares (o que 6*P conta):      {100*lin/total_mac:5.1f}% dos MACs")
print(f"  NAO-lineares (atencao+vocab):    {100*(1-lin/total_mac):5.1f}%")
print()
print("=" * 74)
print("ONDE ISSO MANDA INVESTIR")
print("=" * 74)
print("  1. A ATENCAO e' o maior componente isolado, e o seu custo e' ~2*T*HD por")
print("     par. Ela cresce com T e depende de HD, nao de d_model -- e' por isso")
print("     que head_dim 16 (contra 128 do campo) e' o gargalo medido: 9,8 GFLOP/s")
print("     contra 74,3 com HD=128. Otimizar LINEARES nao toca nisso.")
print("  2. A CABECA DE VOCAB e' o segundo nao-linear: V*d por token. Com V=8192")
print("     e d=96 ela pesa muito. Corte de vocabulario seria um ganho estatico,")
print("     mas custa qualidade (e' a H2 neologismo).")
print("  3. O estimador esta' CERTO para o alvo de 1B (la' os lineares dominam e")
print("     6*P vale), e ERRADO para o nosso regime. Ou seja: a resposta a 'onde")
print("     investir' e' DIFERENTE em cada escala, e o nosso regime nao e' o alvo.")
print()
print("  Leitura operacional: no nosso tamanho, o investimento e' ARQUITETURA")
print("  (head_dim/atencao) e verba de vocabulario, nao kernel de GEMM linear.")
print("  Na escala de 1B inverte, e la' 6*P e' a conta certa.")
