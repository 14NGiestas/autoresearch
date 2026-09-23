# /// script
# dependencies = []
# ///
"""A taxa de um passo por tamanho de modelo, MEDIDA no mesmo setup.

fermi, OMP=8, 12 camadas, batch 1, 1024 tokens por passo, 40 passos por medida.
Cada linha vem de um job: d96 e d768 do 180, d216 e d360 do 177.

Por que isto importa: eu estimava o tempo com FLOP/s constante por tamanho. Nao
e'. O d96 corre a 3 por cento do teto da CPU e o d768 a 25. Um modelo pequeno
gasta o passo em overhead e em memoria, nao em contas.
"""

# arch, params, wall_s (40 passos)
M = [
    ("d96",  2.75e6,  42.6776),
    ("d216", 9.3e6,   29.9919),
    ("d360", 26.0e6,  76.5072),
    ("d768", 88.0e6, 167.1964),
]
TETO_CPU = 517e9

print(f"{'arch':6s} {'params':>8s} {'s/passo':>9s} {'tok/s':>8s} {'GFLOP/s':>9s} {'% teto':>7s}")
for n, P, w in M:
    sp = w/40.0; tok = 1024.0/sp; gf = tok*6.0*P
    print(f"{n:6s} {P/1e6:7.2f}M {sp:9.3f} {tok:8.0f} {gf/1e9:9.1f} {100*gf/TETO_CPU:6.1f}%")
print()
P360, tok360 = 26.0e6, 1024.0/(76.5072/40.0)
CH = 20*P360
print(f"=== o d360: {tok360:.0f} tok/s medidos")
print(f"  Chinchilla (20 tok/param) = {CH/1e6:.0f}M tokens")
print(f"  logo {CH/tok360/86400:.1f} dias numa maquina de 8 threads")
print(f"  e num dia cabem {86400*tok360/1e6:.1f}M tokens = {86400*tok360/(7812*1024):.1f} epocas do pool de 8,0M")
print()
print("=== o teto de cada maquina, para comparar com o medido")
print(f"  CPU, 8 threads:  {TETO_CPU/1e9:.0f} GFLOP/s")
print(f"  iGPU residente:  914 GFLOP/s (medido no resident_layer)")
print(f"  o d360 usa {100*84e9/TETO_CPU:.0f} por cento da CPU e {100*84e9/914e9:.0f} por cento da iGPU")
