# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.2"]
# ///
"""O GFLOP/s do sgemm por FORMA, que e' onde o passo perde o tempo.

Por que isto existe: a atencao e' 4,4 por cento do passo do d360, e o softmax
2,7. O resto esta' nas camadas lineares, que correm a ~84 GFLOP/s contra um teto
de 517. Este script mede o sgemm nas formas REAIS do d360 e do d768, com varios
numeros de threads, para separar efeito de forma de efeito de threading.
"""
import os, sys, time
import numpy as np

def bench(M, N, K, threads, reps=20):
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = A @ B                      # aquecimento
    t0 = time.perf_counter()
    for _ in range(reps):
        np.matmul(A, B, out=C)
    t = (time.perf_counter() - t0) / reps
    return 2.0 * M * N * K / t / 1e9

# as formas do d360 (d=360, ffn=1440, 1024 tokens) e do d768
FORMAS = [
    ("d360 QKV  360x360 x1024", 1024, 360, 360),
    ("d360 O    360x360 x1024", 1024, 360, 360),
    ("d360 up   1440x360 x1024", 1024, 1440, 360),
    ("d360 down 360x1440 x1024", 1024, 360, 1440),
    ("d768 up   3072x768 x1024", 1024, 3072, 768),
    ("micro    768x768 x2048",  2048, 768, 768),
]
print(f"{'forma':28s} " + "".join(f"{f'T={t:<7d}'}" for t in (1, 2, 4, 8, 16)))
for nome, M, N, K in FORMAS:
    linha = f"{nome:28s} "
    for th in (1, 2, 4, 8, 16):
        g = bench(M, N, K, th)
        linha += f"{g:8.1f} "
    print(linha)
print()
print("=== e o passo do d360, somando as formas reais")
# por camada e por token: QKV 3*360*360, O 360*360, up 1440*360, down 360*1440
macs_token = 3*360*360 + 360*360 + 1440*360 + 360*1440
flops_fwd = 2.0 * macs_token * 1024 * 12          # 12 camadas
flops_step = 3.0 * flops_fwd                      # fwd + bwd ~ 3x
print(f"  lineares: {flops_fwd/1e9:.1f} GFLOP no forward, {flops_step/1e9:.1f} no passo")
for th in (1, 4, 8, 16):
    g = bench(1024, 1440, 360, th)
    print(f"  com {th:2d} threads e a forma up: {g:6.1f} GFLOP/s -> passo linear {flops_step/g:6.3f} s")
