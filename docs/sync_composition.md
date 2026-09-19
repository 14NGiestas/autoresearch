# sync_composition — o que custa sincronizar (K=4, 3M params, CPU, τ=122)

Nota de trabalho. Números são bpb de holdout (100 linhas, fixas) do modelo 3M
(`train_run`, attn BLAS), mesmo init (`/tmp/mix/init3m`) e mesmo pool
(`/tmp/mix/rows_f0.npy`) salvo quando dito. Driver: `scripts/fedavg_rounds.py`.
**Estado: as conclusões abaixo já incorporam duas correções nossas** (atribuição
do ganho do outer; mecanismo do reset) — ver as seções marcadas CORRIGIDO.

## O instrumento

| item | valor |
|---|---|
| workers | K=4 (4 processos `train_run`) |
| sincronização | média dos pesos a cada τ passos/worker |
| orçamento | 1952 passos/worker = 16 rodadas a τ=122 (o "wall-clock") |
| compute | K × 1952 = 7808 passos de worker |
| régua parede | run único de 1952 passos = **2,44475** (reproduzido em 2,44476) |
| régua compute | run único de 7808 passos = **2,12000** |
| one-shot | 1 rodada de τ=1952 = 2,50360 (**pior que não compor**) |

## O que fazer com o estado do otimizador (tabela 3×2)

| K=4, τ=122, 1952 passos/worker | sem outer | com outer Nesterov lr=0,7 |
|---|---|---|
| momentos **mediados** entre workers | 2,69018 | 2,42053 |
| momentos **por worker** (sem média) | **2,68616** | **2,42092** |
| momentos **zerados** (reset) | **2,49459** | **2,33183** |

**CORRIGIDO — o veneno é estado estale, não a média.** Mediar os momentos vs
manter cada worker com os seus é *indiferente* (−0,004 e +0,0004, dentro do
ruído). O que importa é não carregar estado desalinhado com os pesos pós-média:
reset ganha por 0,19 (sem outer) e 0,09 (com outer). (Nossa primeira leitura —
"a média de momentos entre workers é o veneno" — está **refutada**.)

## De onde vem o ganho do outer (controle K=1)

| K=1, τ=122, 16 rodadas (mesmo compute) | bpb |
|---|---|
| momentos carregados | 2,71348 |
| + outer Nesterov 0,7 | **2,49638** |

**CORRIGIDO — ~80% do outer é "otimizador melhor", não conserto de sync.** Sem
composição nenhuma o outer vale −0,217; com K=4 vale −0,270. A comparação justa
deixa de ser "composto vs máquina única pura" e passa a ser **"composto+outer
(2,33183) vs máquina única+outer (2,49638)" = 0,16 bpb a favor de compor**, no
mesmo wall-clock (agora com o mesmo otimizador dos dois lados).

## Resíduo sem explicação: a estrutura de rodadas

K=1 com **1 rodada** bate a referência antiga na 5ª casa (2,44476 vs 2,44475) →
o encanamento é neutro. Mas K=1 com **16 rodadas** custa **+0,27** (2,71348), e
depois do outer ainda sobra **+0,05** (2,49638 vs 2,44476). Candidatos: warmup
linear de 2 passos que é *run-relative* e reinicia a cada invocação (32 de 1952
passos), estado do Adam, ou efeito de processo novo. **Job 140** (τ × nº de
rodadas a passos fixos, + um arm sem momentos) testa se o custo é **por
invocação**. Enquanto isso: todos os arms compartilham esse custo, então as
comparações *entre arms* valem; a comparação com a régua de 1 rodada é a que
precisa de qualificação.

## Frequência de sync: satura (e não é lr errado)

| τ | syncs | lr_outer 0,4 | **0,7** | 1,0 |
|---|---|---|---|---|
| 488 | 4 | — | 2,55000 | (1,5 explode) |
| 244 | 8 | em curso | — | — |
| 122 | 16 | — | **2,33183** | — |
| 61 | 32 | 2,38053 | **2,32519** | 2,38800 |

Em τ=61, 0,7 é **mínimo local claro** (os dois vizinhos piores) → o achatamento
**não** é lr mal ajustado. Com o melhor lr, τ=61 empata com τ=122 (−0,007):
**sincronizar o dobro não compra nada**, e o custo de comunicação (16 syncs ≈
4 KB/token) não é a restrição.

## K: o ganho satura

| K (τ=122, 16 rodadas, 1952 passos/worker) | bpb | compute (passos de worker) | fração do ideal |
|---|---|---|---|
| 4 | 2,33183 | 7.808 | 35% |
| 8 | **2,32762** | 15.616 | 28% |

K=8 empata com K=4 com **o dobro do compute**, e o delta é minúsculo em todos os
pontos comparáveis. **Correção de interpretação:** o lote efetivo por
*atualização* **cresce** com K (cada update corresponde a K·τ·T tokens) — o que
satura é a qualidade por token quando se troca *atualizações* por lote. Ou seja:
compor gasta atualizações, e este modelo precisa de muitas (a régua de
7.808 updates vence a de 16 updates com os mesmos tokens).

## O caminho que sobra: o passo

Medido no log do worker (122 passos): **124.928 tokens em 68,5 s = 1.823
tokens/s** com ~8 threads = **30 GFLOP/s ≈ 3,8 GFLOP/s por core**, porque B=1
faz cada GEMM minúsculo (M=1024, N=K=d=96 → 0,019 GFLOP). Composição deu 1,35×;
eficiência de passo tem **3–5×** na mesa (hyp_c3aed6, **job 143**).

Confundimento achado de quebra: cada worker usa ~8 threads de BLAS, mas o job é
alocado com `-c 8` e o driver lança K=4 workers em paralelo → **oversubscrição**.
A comparação por *tokens/worker* segue justa; a de wall-clock precisa de
`OMP_NUM_THREADS` por worker (ou `-c 32`).

## Aberto

| item | onde |
|---|---|
| custo por invocação (warmup/estado)? | job 140 |
| os 3 arms decisivos num 2º sorteio de dados | job 144 |
| τ=244 em lr 0,7 e 1,0 (o 138 foi cortado no limite de 4 h) | job 145 |
| eficiência de passo (B ∈ {1,4,16,64}) | job 143 |
| K=2 e K=16 (fechar a curva em K) | a agendar |
| 2–3 sementes por arm (ruído estimado ±0,017) | a agendar |

## Refs

- `docs/tier_probe.md` — a outra ponta (memória/energia).
- `hep/` — `hyp_ea9457` (esta linha), `hyp_8f9016` (saturação em K),
  `hyp_c3aed6` (eficiência de passo).
- `scripts/fedavg_rounds.py` — driver; `docs/fortran_gpt.md` — a biblioteca.
