# sync_composition — o que custa sincronizar (K=4, 3M params, CPU, T=122)

Nota de trabalho. Todos os números são bpb de holdout (100 linhas, fixas) do mesmo
modelo 3M treinado com Orquestra/`train_run` (attn BLAS), mesmo init
(`/tmp/mix/init3m`) e mesmo pool (`/tmp/mix/rows_f0.npy`, 11797×1025 int32), salvo
quando dito. Driver: `scripts/fedavg_rounds.py`.

## O instrumento

| item | valor |
|---|---|
| workers | K=4 (4 processos `train_run`, 1 core cada, `-c 8`) |
| sincronização | a cada τ passos/worker; média aritmética dos pesos (fedavg) |
| orçamento | 1952 passos/worker = 16 rodadas a τ=122 (o "wall-clock") |
| compute | K × 1952 = 7808 passos de worker (4× o wall-clock) |
| régua 1 (parede) | run único de 1952 passos no pool inteiro = **2,44475** |
| régua 2 (compute) | run único de 7808 passos no pool inteiro = **2,12000** |
| one-shot | 1 rodada de τ=1952 (= fundir no fim, sem laço) = 2,50360 |

## O 2×2 (o achado)

Duas coisas que a gente tratava como uma só ("fundir pesos estraga"):

| τ=122, K=4, 1952 passos/worker | momentos carregados | momentos zerados |
|---|---|---|
| **sem outer** | A: 2,69018 | C: 2,49459 |
| **com outer Nesterov (lr=0,7)** | D: 2,42053 | **B: 2,33183** |

Decomposição:

| efeito | Δ |
|---|---|
| **não carregar/mediar o estado do Adam** (A→C) | **−0,196** |
| **otimizador externo no laço externo** (A→D) | **−0,270** |
| os dois juntos (A→B) | −0,358 (soma seria −0,466 → **sub-aditivos**, ~0,11 comuns) |

Leituras:

1. **A receita composta bate a máquina única no mesmo wall-clock**: B = 2,33183 vs
   2,44475 = **×1,08 em perplexidade**. Até D, que é só o outer, já bate (2,42053).
2. E ainda está **0,212 atrás do ideal de mesmo compute** (2,120) → capturamos
   **~52 %** do ganho paralelo disponível. Essa fração, em função de K, é a pergunta
   que decide a meta de 32 máquinas.
3. O reset dos momentos sozinho (C) é **uma linha de código** e vale −0,196: leva de
   "pior que fundir no fim" (A = 2,690 > one-shot 2,504) a "empatado com fundir no
   fim" (C = 2,495), mesmo sincronizando 16 vezes.

## Frequência de sync: satura

| τ | syncs | final | nota |
|---|---|---|---|
| 488 | 4 | 2,55000 | lr_outer ajustado aqui |
| 122 | 16 | 2,33183 | |
| 61 | 32 | **2,32519** | empate (−0,0066) |

Mas **na metade da curva** τ=61 era uniformemente melhor que τ=122 (mesmos passos por
worker): −0,266 em 244 passos, −0,055 em 976, −0,007 em 1952. Ou seja, as curvas
**convergem**: a vantagem de sincronizar mais cedo não sobrevive ao fim do orçamento.
Prático: **τ=122 (16 syncs, ~4 KB por token) é o ponto útil**; 32 syncs dobra a
comunicação por 0,007.

Cuidado metodológico: com o outer, **τ e lr_outer não são separáveis** (mais sync =
mais passos do outer). O lr=0,7 foi ajustado a τ=488; o bracket 0,4/1,0 em τ=61 e
0,4/0,7/1,0 em τ=244 está na fila para separar "saturou" de "lr errado".

## O que a literatura já sabe (não reivindicar)

- **DiLoCo não reseta o estado interno**: o paper de scaling laws (NeurIPS 2025) diz
  que a diferença para FedOpt é que *"the replicas maintain their inner optimizer
  state across rounds"*. Nosso A **medeia** o estado entre workers; C/D testam não
  fazê-lo.
- **O reset é conhecido em FL**: *Local Adaptivity in Federated Learning* (2106.02305)
  propõe exatamente "restarting the update of client optimizer states at the beginning
  of each round"; *FedAdamW* (AAAI) relata o oposto (reset atrasa a convergência).
- **"optimizer-state mismatch ... amplify drift across rounds"** já é nomeado
  (FedGaLore, 2602.01746).
- **Single-worker DiLoCo bate AdamW mesmo sem distribuição**: então parte do nosso
  −0,270 do outer pode ser *otimizador melhor*, não conserto de sync. Controle K=1
  com/sem outer em 3 pools está na fila — **se ganhar em K=1, a atribuição muda**.

## Controles (fila)

| job | o que |
|---|---|
| 137 | K=1 com/sem outer × 3 pools (f0 + 2 permutações; não há flag de seed, a réplica honesta é outro sorteio de dados); K=4 com momentos **por worker** (sem media) ± outer |
| 138 | bracket lr_outer × τ |
| 136 | K=8 a τ=122 (o ganho escala com K?) |

`--carry-per-worker` é novo em `scripts/fedavg_rounds.py` e foi validado bit-exato:
estado do seed = estado do próprio worker (diff 0,0), pesos = média (diff 0,0),
pesos ≠ worker (0,068).

## Refs cruzadas

- `docs/mpi_ddp.md` — o caminho multi-máquina (ssh/NAT, invariante bit-exato).
- `hep/` — registro com hash chain; hipótese `hyp_ea9457` (sync/composição).
- `STATUS.md` — estado corrente.
