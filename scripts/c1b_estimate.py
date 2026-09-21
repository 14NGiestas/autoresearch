# /// script
# dependencies = []
# ///
"""O C-1B num dia, so' com numeros MEDIDOS nesta bancada.

Tres erros que este script evita, todos cometidos antes nesta sessao:
  1. tratar a taxa de tokens como constante da maquina (ela e' FLOP/s / 6P);
  2. usar a fraccao do teto da CPU (58 por cento) para a iGPU;
  3. usar a taxa da iGPU do caminho por chamada (227 tok/s), que era presa da
     transferencia, para o desenho residente.
A fraccao honesta da iGPU e' a do resident_layer: 914 de 2300 GFLOP/s, 40 por cento.
"""

MEDIDOS = {
    "iGPU residente":  914.4e9,   # resident_layer, 14,091 ms por camada
    "iGPU teto":       2300e9,    # gpubench, rocBLAS sgemm
    "CPU treino":      299e9,     # job 165, 88M, 566 tok/s
    "CPU teto":        517e9,     # microbench de GEMM
}

def dias(P, flop_s):  return 20.0 * P / (flop_s / (6.0 * P)) / 86400.0
def maior(flop_s, d=1.0): return (d * 86400.0 * flop_s / 120.0) ** 0.5
def anos(P, flop_s):  return dias(P, flop_s) / 365.25

print("=== C-1B: 1e9 params, 2e10 tokens, 1,2e20 FLOPs")
for n, f in MEDIDOS.items():
    print(f"  {n:16s} {f/1e12:5.2f} TFLOP/s: {anos(1e9, f):7.2f} anos  ({dias(1e9,f):9.0f} dias)")
print(f"  {'CPU + iGPU':16s} {sum(MEDIDOS.values())/1e12:5.2f} TFLOP/s: {anos(1e9, MEDIDOS['CPU treino']+MEDIDOS['iGPU residente']):7.2f} anos")
print()
print("=== o maior Chinchilla que cabe num dia")
for n, f in MEDIDOS.items():
    P = maior(f); print(f"  {n:16s}: {P/1e6:6.1f}M params")
print()
print("=== e o 1B contra um dia, em multiplos")
alvo = anos(1e9, MEDIDOS["iGPU residente"])
print(f"  uma iGPU:  {alvo:.2f} anos = {alvo*365.25:.0f} dias = {alvo*365.25:.0f} vezes um dia")
print(f"  para o 1B caber num dia seriam precisas {alvo*365.25:.0f} iGPUs")
print()
print("=== os nossos pontos, na iGPU residente")
for n, P in [("d96",2.75e6),("d216",9.3e6),("d288",14.5e6),("d360",26e6),("d768",88e6)]:
    print(f"  {n:6s} {P/1e6:6.2f}M: {dias(P, MEDIDOS['iGPU residente']):7.3f} dias de Chinchilla")
