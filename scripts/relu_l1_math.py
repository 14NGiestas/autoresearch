# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.2"]
# ///
"""A matematica do relu/sum(relu), verificada por diferencas finitas.

O forward, por linha causal:
    s_i  = score i (ja com escala e cap)
    r_i  = max(s_i, 0)                    (relu)
    S    = sum_j r_j                      (L1 da linha)
    P_i  = r_i / S                        (a normalizacao que a literatura diz ser
                                           o componente critico, e nao a exponencial)
    y    = P . v

O backward recebe dP e devolve ds. Com o softmax, ds = P*(dP - sum(P*dP)). Com
relu puro sobre T, ds = dP/T (constante). Com relu sobre S, o denominador DEPENDE
de todos os scores, entao entra a regra do quociente:

    dP_i = d(r_i)/S - r_i * dS / S^2
    dS   = sum_j d(r_j) = sum_j r'_j ds_j
    d(r_i) = r'_i ds_i

    ds_k = r'_k * [ dP_k / S  -  (sum_i r_i dP_i) / S^2 ]

que sao DOIS termos: o do numerador e o do denominador. O segundo e' o que falta
se alguem portar so' o primeiro, e e' exactamente a classe de erro que o porteiro
de FD apanhou no relu/T (dq errado por 1,2).
"""
import numpy as np

rng = np.random.default_rng(7)

def fwd(s, eps=0.0):
    r = np.maximum(s, 0.0)
    S = r.sum()
    P = r / S if S > 0 else np.zeros_like(r)
    return P

def bwd_analytic(s, dP):
    r = np.maximum(s, 0.0)
    S = r.sum()
    rp = (s > 0).astype(float)          # derivada do relu
    term1 = dP / S                       # do numerador
    term2 = (r @ dP) / (S * S)           # do denominador
    return rp * (term1 - term2)

def check(n=9, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n) * 2.0
    dP = rng.normal(size=n)
    h = 1e-6
    fd = np.zeros(n)
    for i in range(n):
        sp = s.copy(); sp[i] += h
        sm = s.copy(); sm[i] -= h
        fd[i] = (fwd(sp) @ dP - fwd(sm) @ dP) / (2 * h)
    an = bwd_analytic(s, dP)
    err = np.max(np.abs(fd - an))
    # e o que aconteceria se alguem esquecesse o termo do denominador
    only1 = (s > 0) * (dP / np.maximum(np.maximum(s, 0).sum(), 1e-30))
    err1 = np.max(np.abs(fd - only1))
    return err, err1

print(f"  {'caso':<28} {'erro do backward completo':>26} {'erro sem o termo do denom.':>28}")
for k in range(5):
    e, e1 = check(seed=k + 1)
    print(f"  {('caso ' + str(k+1)):<28} {e:>26.3e} {e1:>28.3e}")
print()
print("  se a coluna 2 e' ~1e-10 e a coluna 3 e' grande, a derivacao esta' certa e o")
print("  termo do denominador e' de facto necessario.")
