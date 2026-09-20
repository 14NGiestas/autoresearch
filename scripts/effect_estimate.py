#!/usr/bin/env python3
"""Previsao antes da medida: quanto um efeito deveria mover o numero.

A regra, e ela tem uma razao pratica: antes de enfileirar um job, escreva quanto
o efeito DEVERIA valer. Uma previsao de ordem de grandeza decide se o job vale a
fila, e -- mais importante -- torna o erro da nossa propria previsao um dado.
Sem previsao declarada antes, uma medida so' confirma o que ja' se acredita.

Cada entrada tem quatro campos obrigatorios:
  modelo     a conta, em palavras, para qualquer um refazer
  previsao   o numero esperado, com a unidade
  falsifica  o que a medida tem que mostrar para a previsao estar ERRADA
  medida     o valor medido, quando existir (vazio = ainda na fila)

O script calcula o que da' para calcular (nada digitado) e imprime a coluna de
erro da previsao quando a medida chegou. Um laboratorio que nao mede o erro das
proprias previsoes nao sabe se esta' pensando ou adivinhando.

Uso:  .venv-numpy/bin/python3 scripts/effect_estimate.py
"""
import sys

# Relogio da maquina. fermi: ~3 GHz efetivo com AVX2. A conta e' de ordem de
# grandeza, entao 20% de erro aqui nao muda a conclusao.
CLOCK_GHZ = 3.0

# Ciclos por chamada de tanh, por elemento, com ILP entre as chamadas da linha.
# O polinomio tem 8 termos e nenhum desvio; o intrinseco tem desvios e uma
# chamada de libm. Sao estimativas: o job 157 mede o resultado.
CYCLES = {
    "fast_tanh (polinomio, 8 termos)": 4.0,
    "tanh intrinseco (libm)": 40.0,
}


def tanh_calls(batch, heads, layers, ctx):
    """Uma chamada por par de scores causal: T*(T+1)/2 por (batch, cabeca, linha)."""
    pairs_per_row = ctx * (ctx + 1) / 2
    return batch * heads * layers * pairs_per_row


def main():
    # ---- caso 1: fast_tanh contra tanh exato, no regime com cap ligado
    B, H, L, T = 1, 6, 12, 1024
    step_s = 1.211                     # medido: job 154, arm h6_kv2 (d96)
    calls = tanh_calls(B, H, L, T)

    print("=" * 76)
    print("CASO 1: o fast_tanh move o passo, no regime COM cap ligado?")
    print("=" * 76)
    print(f"  modelo: chamadas = B*H*L*T*(T+1)/2")
    print(f"  {B}*{H}*{L}*{T}*({T}+1)/2 = {calls:,.0f} chamadas por passo")
    print(f"  passo medido (sem cap): {step_s:.3f} s  (job 154, d96 h6 kv2)")
    print()
    print(f"  {'implementacao':<34} {'ciclos/cham':>11} {'s/passo':>9} {'do passo':>9}")
    pred = {}
    for name, cyc in CYCLES.items():
        sec = calls * cyc / (CLOCK_GHZ * 1e9)
        pred[name] = sec
        print(f"  {name:<34} {cyc:>11.1f} {sec:>9.3f} {100*sec/step_s:>8.0f}%")
    d = pred["tanh intrinseco (libm)"] - pred["fast_tanh (polinomio, 8 termos)"]
    print()
    print(f"  vao previsto entre os dois: {d:.3f} s por passo = {100*d/step_s:.0f}% do passo")
    # P ~ sqrt(threads x tempo): um passo mais lento encolhe o modelo alcancavel
    # na mesma janela pela raiz do fator.
    slow = (step_s + d)/step_s
    print(f"  efeito na JANELA: o mesmo modelo leva {slow:.2f}x mais tempo, ou")
    print(f"                    o modelo alcancavel na mesma janela cai a {1/slow**0.5:.2f}x")
    print()
    print("  previsao: com cap e fast_tanh o passo cresce ~4%; com o tanh exato,")
    print(f"            ~{100*pred['tanh intrinseco (libm)']/step_s:.0f}%. A janela move ~1,4x.")
    print("  falsifica: se |s/step(fast) - s/step(exato)| no job 157 ficar abaixo de")
    print("            0,05 s, a previsao erra por 10x e o fast_tanh nao e' alavanca.")
    print("  medida:   (vazio -- job 157 na fila)")
    print()

    print("=" * 76)
    print("CASO 2: o fast_tanh move o passo, no regime SEM cap (o de hoje)?")
    print("=" * 76)
    print("  modelo: o cap desligado retorna antes de qualquer tanh (cap <= 0).")
    print(f"  chamadas de tanh por passo: 0")
    print(f"  previsao: 0,000 s = 0% do passo. Nao ha o que medir.")
    print("  falsifica: qualquer diferenca entre s/step com e sem o binario exato.")
    print("  medida:   ja' implicita -- nenhuma chamada e' feita.")
    print()

    print("=" * 76)
    print("ERRO DAS NOSSAS PREVISOES (a coluna que da' sentido a este script)")
    print("=" * 76)
    print("  caso 1: aguardando o job 157")
    print("  caso 2: sem erro possivel (zero chamadas e' identidade)")
    print()
    print("  Regra: a proxima previsao entra aqui ANTES do job, com o criterio de")
    print("  falsificacao escrito. Depois do job, o valor medido entra ao lado, e o")
    print("  erro fica visivel. Previsao sem falsificacao nao e' previsao, e' torcida.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
