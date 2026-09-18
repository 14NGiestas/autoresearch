# MPI: data-parallel de verdade (sync grad / sync delta)

Data: 2026-09-18. Arquitetura compilada: **d216** (D=216, nh=6, nkv=2, nl=12,
V=8192, T=1024), a mesma de `src/build/arch_d216_*`.

> **Como as evidencias foram rodadas — e o INCIDENTE que isso causou.**
> `jobs/fermi_mpiddp.sbatch` (job 107) foi submetido, mas a fila `debug` estava
> saturada pelos jobs do proprio usuario (103 rodando, 104/105/106 na frente), e
> as evidencias deste documento sairam de execucoes em **foreground** no mesmo no
> (`fermi` = no de login E de computo). Isso foi um erro grave: as 07:18 de
> 2026-09-18, com 2 ranks do `train_ddp` (760-845 MB/rank) em paralelo com o job
> 103 (`scal65`), a fermi (15 GB) entrou em **global_oom** e o kernel matou o
> `train_run` alheio (RSS 3.5 GB), perdendo os 2 ultimos checkpoints de uma curva
> de 7h40.
>
> **Regra dura a partir daqui (esta implementada no codigo, nao so' comentada):**
> `mpirun`/`mpiexec` **nunca** fora de um job do Slurm, **sempre com `--mem`
> explicito** (4 GB para `-n 2`/smoke, 6 GB para `-n 4`, 10 GB para `-n 8`) e
> **nunca em paralelo com um run longo**. `mpi/tests_ddp.sh` e `mpi/tests_tax.sh`
> recusam rodar sem `SLURM_JOB_ID` (exit 2), e `jobs/fermi_mpiddp.sbatch` carrega
> `--mem=4G` + um portao que aborta se houver qualquer outro job RUNNING do
> usuario. O job 107 (que ja' estava na fila) ganhou o teto via
> `scontrol update jobid=107 MinMemoryNode=4096` em vez de resubmissao — mas com
> `RealMemory=1 MB` isso o deixou insatisfazivel (`BadConstraints`), entao o teto
> foi revertido para 0 e o **job 107 foi cancelado e resubmetido como 114** com o
> wrapper novo (escopo systemd + portao que re-enfileira em vez de abortar), por
> autorizacao do supervisor.
>
> Os logs citados aqui estao em `mpi/scratch_20260917/final_ddp.log` e
> `final_tax.log` (rodadas antes da regra; os numeros sao validos, o *modo de
> execucao* nao se repete).

## 1. Por que

A emulacao de varios workers que ja temos (local SGD por rodadas, checkpoint
trocado por `scp`) testa so **tau grande** — ela existe porque com tau grande a
sincronizacao e' barata de *implementar* (um arquivo por rodada) e impossivel de
*medir* com precisao (o "worker" de cada rodada e' um processo serial).

A matematica diz outra coisa: o unico ponto onde o imposto de sincronizacao e'
**zero** e' `tau = 1`, porque a media dos gradientes de K lotes e' exatamente o
gradiente de um lote K vezes maior. Medir isso exige os K processos
sincronizando *de verdade* a cada passo — ou seja, MPI. E' o que este trabalho
destrava:

| | antes | agora |
|---|---|---|
| topologia | rodadas seriais + `scp` de checkpoint | K ranks MPI |
| tau measuravel | ~epochs | 1, 2, 3, ... |
| sync | arquivo em disco (GB por rodada) | `MPI_Allreduce` de 38 MB |
| invariante testavel | "os pesos batem depois do merge" | **bit a bit**, passo a passo |

## 2. O que foi feito

**(1) Refatoracao pura em `src/lib/fortran_train.f90`.** As 8 chamadas
`apply_group`/`apply_opt` que estavam no fim de `train_step` viraram a
subroutine **publica** `apply_update(M, S, GR, tstep, lr, b1, b2, beps, wd,
use_muon, lr_muon, G)`. `train_step` agora chama so' ela; a ordem, os kernels e
os argumentos sao os mesmos (nenhuma matematica mudou).

> Desvio do pedido, explicito: `apply_update` recebe **um argumento a mais,
> `G` (dims_t), e ele e' OPTIONAL**. Motivo: a geometria `(rows, cols)` por
> grupo so' e' lida pelo ramo **Muon** (`adamw_step` anda sozinho sobre o
> buffer flat); sem ela o Muon nao tem como montar a vista 2D. Com
> `present(G) = .false.` o caminho AdamW roda igual e `use_muon=.true.` aborta
> alto em vez de fazer Muon errado. `train_step` e o app MPI sempre passam `G`.

**(2) App MPI novo, em projeto fpm separado** — nao encosta em `src/build/arch_*`
(jobs na fila resolvem o binario na hora da execucao):

```
mpi/fpm.toml              projeto fpm proprio (mpif90, --build-dir build)
mpi/app/train_ddp.f90     o trainer data-parallel
mpi/tests_ddp.sh          invariantes (1 rank x 2 ranks, byte a byte)
mpi/tests_tax.sh          quanto custa tau=1 em wall-clock
jobs/fermi_mpiddp.sbatch  os dois scripts acima dentro do SLURM
```

Flags: `--weights --rows --out --nsteps --lr --ntrain --start_row --bytes
--save_every --sync_every H --sync grad|delta --data slice|rotate` (+ `--chunk`,
`--save_all`, `--attn`, `--opt adam|muon`, `--t0`, `--nval`,
`--val_every`, `--ckpt-format npy|st|both`, `--sync_log`).

Semantica, como pedida:
- `--data slice`: rank r le `[r*chunk, (r+1)*chunk)` do pool, em ciclo
  (`offset = mod(rank*chunk + mod(k-1, chunk), ntrain)`). `--chunk ntrain` faz
  **todos os ranks lerem o mesmo pool** (e' assim que o invariante e' testado).
- `--data rotate`: rank r le a linha `(passo*size + r) mod ntrain` — lotes
  disjuntos por passo, pool inteiro coberto por passo.
- `--sync grad`: `MPI_Allreduce(SUM)` em **cada** array de `GR`, divide por
  `size`, e so' entao `apply_update`. Todos os ranks aplicam o MESMO update.
- `--sync delta`: copia dos pesos no inicio do bloco, H passos locais, depois
  `Allreduce(SUM)` do delta `M - M0`, divide por `size`, `M = M0 + delta/size`.
- Checkpoint so' no rank 0 (`--save_all 1` grava `out/rank_<r>/...`).

## 3. Build (exato)

```bash
cd /home/pauli/autoresearch
nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich -c bash -c '
  cd mpi
  FPM_FC=mpif90 fortran-fpm build --profile release \
      --flag "-march=native -ffast-math" --build-dir build
'
# binario: mpi/build/mpif90_6F9FB35139E0E254/app/train_ddp
```

Por que essas escolhas (todas aprenderam batendo):
- `FPM_FC=mpif90`: o modulo `mpi` vem do MPI instalado, nao de um fonte do
  projeto → `external-modules = ["mpi"]` em `mpi/fpm.toml`, e o compilador e' o
  wrapper do MPICH.
- `[build] link = ["openblas", "mpi"]`: o `link` do manifesto de `src/` **nao**
  propaga para o pacote dependente; sem repetir, `undefined reference` no
  `sgemm_` e nos simbolos do MPI.
- `[[executable]]` explicito: o auto-scan de `app/` do fpm 0.13 nao
  dispara num pacote **sem** `[library]` ("Neither library nor executable
  found").
- `path = "../src"` (e nao `".."`): o manifesto `fortran_gpt` vive em `src/`,
  nao na raiz do repo. Este projeto fica na raiz (`mpi/`) exatamente para nao
  entrar no source-scan do `fortran_gpt` (`src/`, `app/` do outro projeto).
- `mpi/build/cache.toml` aponta os deps ja' baixados (`stdlib`, `safetensors`)
  em `mpi/build/dependencies/` — sem ele o fpm re-baixa tudo.
- `-march=native` e' **ignorado** dentro do nix (`NIX_ENFORCE_NO_NATIVE`), como
  no build de `arch_d216`. Os flags efetivos sao:
  `-march=native -ffast-math -O3 -Wimplicit-interface -fPIC -fmax-errors=1
  -funroll-loops -fcoarray=single -fopenmp`.

Baseline obrigatorio antes/depois (projeto principal, verde): `fortran-fpm test`
em `src/` → **`===0 failures ===`** nos dois programas de teste.

## 4. Invariante: 2 ranks x 1 rank, byte a byte

```
mpirun -n 2 mpi/build/*/app/train_ddp \
  --weights /tmp/w_p2soup --rows /tmp/mix/rows_dev.npy --out .../inv2 \
  --nsteps 3 --lr 1e-3 --ntrain 8 --chunk 8 --save_every 3 --data slice \
  --sync grad --sync_every 1 --save_all 1 --nval 2 --val_every 3 --bytes .../token_bytes.txt
mpirun -n 1 <mesmo binario> <mesmas flags, sem --save_all>
```
(`--chunk 8` = pool inteiro ⇒ os DOIS ranks leem os mesmos dados ⇒ com
`--sync grad --sync_every 1` a sequencia de operacoes e' a de um run de 1 rank.)

**Diff: zero.** Saida do teste (`mpi/scratch_20260917/final_ddp.log`):

```
=== 2/3. INVARIANTE -n 2 x -n 1 (sync grad, tau=1, chunk = pool inteiro) ===
  fingerprints por rank/passo (-n 2):
    rank 1 step 1 row 0 nll    4.81556 lr    0.00050 fp 9858D14FED582F27 sync 1
    rank 0 step 1 row 0 nll    4.81556 lr    0.00050 fp 9858D14FED582F27 sync 1
    rank 1 step 2 row 1 nll    4.81001 lr    0.00100 fp BDB3264E91588A92 sync 1
    rank 0 step 2 row 1 nll    4.81001 lr    0.00100 fp BDB3264E91588A92 sync 1
    rank 0 step 3 row 2 nll    4.50367 lr    0.00100 fp 3483A613FFB2A821 sync 1
    rank 1 step 3 row 2 nll    4.50367 lr    0.00100 fp 3483A613FFB2A821 sync 1
  fingerprints por passo (-n 1):
    rank 0 step 1 row 0 nll    4.81556 lr    0.00050 fp 9858D14FED582F27 sync 1
    rank 0 step 2 row 1 nll    4.81001 lr    0.00100 fp BDB3264E91588A92 sync 1
    rank 0 step 3 row 2 nll    4.50367 lr    0.00100 fp 3483A613FFB2A821 sync 1
  tree_hash -n 2 rank_0 = f02110e2ed2116a3
  tree_hash -n 2 rank_1 = f02110e2ed2116a3
  tree_hash -n 1        = f02110e2ed2116a3
  -> inv1/step_3 x inv2/rank_0/step_3: 90 arquivos byte-identicos, 0 diferentes
  -> inv2/rank_0/step_3 x inv2/rank_1/step_3: 90 arquivos byte-identicos, 0 diferentes
    bit-identical across ranks: min 0x3483A613FFB2A821 max 0x3483A613FFB2A821 -> T
```

Os 90 `.npy` de cada `step_*` sao **74 de peso** (6 por camada x 12 + `wte` +
`lm_head`) + **16 do Adam** (`adam_m_*`/`adam_v_*`), comparados com `cmp` byte a
byte. As duas relacoes que importam:

* **1 rank x 2 ranks**: 90 identicos, 0 diferentes (`cmp` byte a byte).
* **rank 0 x rank 1**: 90 identicos, 0 diferentes — os ranks ficam
  bit-identicos passo a passo (o `fp` e' igual em cada `step`).

E a matematica do `tau=1` aparece exatamente como prometido: o Allreduce de dois
gradientes identicos e' `g+g = 2g` (exato, so' muda o expoente) e a divisao por
`size=2` desfaz sem arredondamento. Nada de sortudo: e' a mesma sequencia.

Outras combinacoes testadas (mesmo log):

| teste | resultado |
|---|---|
| `--data rotate -n 2` (lotes **disjuntos**, o caso util) | rank0 e rank1 com o **mesmo `fp`** em cada passo; dados lidos de linhas diferentes (`row 0/1`, `2/3`, `4/5`) |
| `--sync delta --sync_every 3 -n 2` | no fim do bloco os **pesos** batem bit a bit (`min == max`); os 16 `adam_*.npy` **diferem** (ver secao 6) |
| `--attn blas -n 2` x `-n 1` | 90 identicos, 0 diferentes (a invariante nao depende do kernel de atencao) |

**Refactor puro, prova independente:** o binario **pre-refactor** de d216
(`src/build/arch_d216_*/gfortran_CC0F834C608B9708/app/train_run`, de 2026-09-17
05:00 — so' leitura, nada foi rebuildado) rodado 1 passo com os mesmos pesos,
linhas, LR e warmup produz os **mesmos 90 `.npy` byte a byte** que
`train_ddp -n 1` depois do refactor:

```
=== 1. REFACTOR PURO: train_run (pre-refactor) x train_ddp -n 1, 1 passo ===
  -> ref_old/step_1 x ref_new/step_1: 90 arquivos byte-identicos, 0 diferentes
```

Ou seja: `apply_update` nao mexeu em um bit — e o app MPI reproduz a aritmetica
do trainer que gerou os runs registrados.

## 5. O imposto de `tau = 1`, medido

> **ATENCAO — kernel.** As medidas desta secao (runs A/B/C/D) foram feitas
> quando o default do `train_ddp` ainda era
> `--attn naive`. O naive e' ~13x mais lento que o blas (`~13 s/passo` contra
> `2.460 tok/s` medidos no `train_run`), entao **96 tok/s de -n 1 e' o kernel
> naive, nao o teto do MPI**. O que continua valendo aqui e' a *razao* entre as
> configuracoes e o custo do coletivo; o nivel absoluto com `--attn blas` esta
> na secao 5b. Desde 2026-09-18 o default do app e' `--attn blas`.

Mesmo computo, mesmo numero de passos, mesmos ranks e threads: so' muda a
frequencia do coletivo (`mpi/tests_tax.sh`, 6 passos, `--data rotate`, 4
threads/rank, kernel naive). A maquina esta' compartilhada com o job 103 do
usuario (8 das 16 CPUs), entao os segundos absolutos sao indicativos.

| run | sync | syncs | wall_s | `allreduce_s` (medido no app) |
|---|---|---|---|---|
| A | grad tau=1 | 6 | 68.59 | 4.257 (6.2% do wall) |
| D | delta tau=3 | 2 | 64.56 | 3.552 |
| B | delta tau=6 | 1 | 62.84 | 0.289 |
| C | 1 rank (sem coletivo) | 0 | 63.69 | 0.029 |

- **Computo util**: C faz 6144 tokens em 63.7 s (96 tok/s). A/D/B fazem 12288
  tokens (2 ranks, lotes disjuntos) em 63-69 s → **179-196 tok/s, ~2x**. Em
  `tau=1` a sincronizacao nao come o ganho de data-parallel.
- **Custo de UM coletivo** (38 MB de payload, 2 ranks/1 no): medido com 1
  thread por rank (`--sync_log 1`, que imprime o tempo de cada `MPI_Allreduce`),
  tempos observados: `0.0149 0.0157 0.0186 0.0212 0.0967 0.3859` s — ou seja
  **~15 ms minimo** (≳2.5 GB/s de payload pelo canal de shm do MPICH), com
  caudas de centenas de ms que sao **contencao de CPU** (o processo
  desescalonado paga wall dentro do coletivo), nao custo do MPI.
- **Imposto de tau=1**: `(A-B)` via wall = **0.96 s/passo** com a maquina
  oversubscribed; `(A-D)/4` (4 syncs a menos) = 0.176 s/sync; em configuracao
  folgada (1 thread/rank, `--sync_log 1`) **0.015-0.02 s/sync**. Ou seja, o
  imposto de tau=1 vai de **~0.1% a ~8%** do passo de ~11 s dependendo de quanta
  CPU o rank realmente recebe: o custo do coletivo e' ~15-20 ms, o resto e'
  fila de CPU. Em 2 nos (rede) isso muda de escala e **nao foi medido** (secao 6).

## 5a. Throughput medido com `--attn blas` (job 113)

Mesma config nos 4 runs: `--data rotate` (lotes **disjuntos**, um por rank por
passo), 4 threads/rank, 20 passos, ntrain=64, d216/T=1024.

| run | ranks | tau | tokens/s (global) | wall_s | `Allreduce` por sync (ms) |
|---|---|---|---|---|---|
| A | 1 | 1 | **1640** | 12,49 | 3,0 (mediana 3,3) |
| B | 2 | 1 | **2759** (1,68x) | 14,85 | 13,3 (mediana 16,1 = 38 MB em shm) |
| C | 1 | 100 | **1695** | 12,09 | 6,7 (1 sync) |
| D | 2 | 100 | **2999** (1,77x) | 13,66 | 21,5 (1 sync) |

- **O `--attn naive` era o gargalo, nao o MPI**: com o default antigo (naive, 1
  thread) medimos 96 tok/s em `-n 1`; com blas e 4 threads/rank sao 1640 tok/s.
- **Imposto de tau=1 no mesmo no**: A x B = 2759/1640 = 1,68x de ganho com 2
  ranks (o maximo seria 2x) e D x B = 2999/2759 → o coletivo por passo custa
  ~8% do tempo de B; o Allreduce de 38 MB mede ~13-16 ms em shm.
- **Achado que vale para o plano multi-maquina**: **2 ranks x 4 threads (2999
  tok/s) batem 1 processo x 8 threads (2460 tok/s, medido no `train_run`)** nesta
  maquina — MPI escala melhor que OpenMP aqui (o OpenMP satura por banda de
  memoria/threads do BLAS, o MPI soma memoria NUMA + BLAS por rank). Isso e'
  argumento direto a favor de distribuir em vez de aumentar threads.

## 5b. Buffer por sync, custo do coletivo e o tau pratico entre maquinas

**O coletivo carrega SO' o gradiente — os momentos nunca entram.** Em
`--sync grad` o unico array que vai para o `MPI_Allreduce` e' o acumulador de
gradiente `GA` (`mpi/app/train_ddp.f90:277`; `allreduce_params` percorre os 8
grupos de `GA`); o estado do Adam (`S`, com `m` e `v`) **nunca** e' passado a
nenhum coletivo. Em `--sync delta` o coletivo leva o delta dos pesos
(`GR = M - M0`, linha 289), tambem 38,04 MB. Os 38,04 MB da tabela abaixo sao
9.510.912 floats = **exatamente o numero de parametros** (o gradiente tem o
tamanho dos parametros) — nao ha' "3x isso por causa de m,v": se os momentos
fossem sincronizados seriam 3 x 38,04 = 114,1 MB por sync.

Por que os momentos nao precisam (e a reducao de 3x ja' esta' feita): os ranks
partem do MESMO checkpoint (pesos **e** `adam_*.npy`), aplicam o MESMO gradiente
medio no MESMO `tstep` e com o MESMO `lr`, entao `m, v` evoluem identicamente
por construcao (`adamw_step` e' funcao determinista de `(m, v, g, t)`). A prova
empirica esta' na secao 4: a comparacao `rank_0 x rank_1` inclui os 16
`adam_*.npy` do checkpoint e deu **90 arquivos byte-identicos, 0 diferentes**
com o coletivo carregando so' o gradiente. (E' tambem por isso que o
`--sync delta` DIVERGE nos `adam_*.npy`: la' os passos sao locais, entao os
momentos saem de caminhos diferentes — o unico caso em que eles precisariam de
sync, e hoje nao sao sincronizados.)

O que reduz trafego, se precisar: (i) `--sync-fp16 bf16|fix` (implementado,
secao 5c: 38,04 -> 19,02 MB, 2x, com o default OFF e a trajetoria bit-exata
intacta); (ii) subir `tau` (a tabela do imposto abaixo); (iii) otimizadores de
estado local que nao sincronizam gradiente (fora do escopo).

**Tamanho de UM sync (este binario, d216/T=1024):** o app faz 8 `MPI_Allreduce`
por sync, um por grupo de `GR` (`--sync grad`) ou do delta de pesos
(`--sync delta` — mesmo tamanho):

| grupo | floats | MB |
|---|---|---|
| `wte` | 1.769.472 | 7,08 |
| `lm` | 1.769.472 | 7,08 |
| `q` | 559.872 | 2,24 |
| `k` | 186.624 | 0,75 |
| `v` | 186.624 | 0,75 |
| `p` | 559.872 | 2,24 |
| `fc` | 2.239.488 | 8,96 |
| `p2` | 2.239.488 | 8,96 |
| **total** | **9.510.912** | **38,04** (36,3 MiB) |

**Custo medido em shm (2 ranks, 1 no):** ~15 ms minimo por sync (o menor tempo
observado de um `MPI_Allreduce` de 38 MB via `--sync_log 1`), com caudas de
0,1-4 s puramente por contencao de CPU. Referencia: ~2,5 GB/s de payload.

**Custo estimado em rede (1 GbE, ~100 MB/s de payload):** 38 MB × 2 trocas (um
allreduce de 2 ranks e' reduce + broadcast) = 76 MB → **0,6-0,8 s por sync**,
contra a estimativa do supervisor de 0,2-1 s para 24 MB (de acordo em ordem de
grandeza). **Cada passo de treino comunica 38 MB para produzir 1024 tokens por
rank: 37 KB de trafego por token.** Esse e' o numero que decide o tau:

| tau | syncs por 1000 passos | trafego fp32 (38,04 MB) | trafego 16 bits (19,02 MB) | imposto fp32 | imposto 16 bits |
|---|---|---|---|---|---|
| 1 | 1000 | 38 GB | 19 GB | +60% a +120% (inviavel) | +38% a +100% |
| 2 | 500 | 19 GB | 9,5 GB | +30% a +60% | +19% a +50% |
| 4 | 250 | 9,5 GB | 4,8 GB | +15% a +30% | +10% a +25% |
| 8 | 125 | 4,8 GB | 2,4 GB | +8% a +15% | +5% a +12% |
| 16 | 63 | 2,4 GB | 1,2 GB | +4% a +8% | +2,4% a +6% |
| 100 | 10 | 0,4 GB | 0,2 GB | +0,6% a +1,2% | +0,4% a +1% |

(imposto = tempo de sync / tempo de passo; sync de 2 ranks em 1 GbE a ~100 MB/s
de payload = 2 x payload do buffer; passo de 0,5-1 s em 4 threads/rank. Coluna
fp32: 38,04 MB -> 0,6-0,8 s de sync. Coluna 16 bits: 19,02 MB -> 0,3-0,4 s.)

Ou seja: **entre maquinas ligadas por 1 GbE o tau pratico fica em 8-16** com o
buffer fp32 (e em **4-8** com o buffer de 16 bits da secao 5c) para um imposto
<=15%; em shm (mesmo no) ele fica em 1-2. Este e' exatamente o teste que
pode ser feito no experimento cross-node (fermi + halfbeast), coordenado pelo
supervisor: a mesma tabela medida com `--sync_log 1` do outro lado decide o tau.
Esses numeros sao **estimativas** (o termo de computo vem de 2.460 tok/s a 8
threads, medido no `train_run`); as medicoes pendentes do job `mpithru` (113)
(`mpi/tests_throughput.sh`, `--attn blas`, 4 threads/rank) fecham o termo de
computo em `-n 1` e `-n 2`. O buffer de 38,04 MB por sync **nao muda** com essas
medicoes (e' o numero de parametros), e o invariante da secao 4 ja' foi provado
com ele (foi esse o coletivo que gerou os 90/90 arquivos identicos).

## 5c. `--sync-fp16 bf16|fix` — 2x menos trafego, DEFAULT OFF

Implementado a pedido do supervisor, **desligado por default**: com
`--sync-fp16 none` (o default) o caminho e' exatamente o de antes — fp32,
bit-exato, o ativo de verificacao da secao 4 intacto. Com `bf16` ou `fix` o
payload do coletivo cai de 38,04 MB para **19,02 MB** (2x):

| modo | o que faz | coletivos por sync |
|---|---|---|
| `none` (default) | `MPI_Allreduce` fp32 dos 8 grupos | 8 (38,04 MB) |
| `bf16` | quantiza o topo do fp32 (RNE no bit 16, 8 bits de mantissa), `Gather` no root, soma em fp32 no root, `Bcast` do resultado requantizado | 2 (19,02 MB + 19,02 MB) |
| `fix` | ponto fixo de 16 bits com escala dinamica (`MAX` de \|g\| por grupo num coletivo de 8 `real64`): pre-escala por `1/nranks`, `MPI_Allreduce(MPI_INT16_T, SUM)` **exato** (a soma cabe no int16 por construcao) | 9 (8 valores + 19,02 MB) |

Nao existe IEEE half em Fortran/MPICH (sem `real16`): `bf16` e' o truncamento de
16 bits do fp32 e `fix` e' ponto fixo — para gradiente, `fix` tem ~15 bits
efetivos de precisao uniforme contra ~8 do `bf16`, e por isso a expectativa e'
de desvio menor (a confirmar com o job `mpiquant`).

**Erro por elemento (teste unitario local `mpi/qprobe.f90`, sem MPI, ~10 ms):**
`bf16` — erro relativo maximo **3,89e-3** (pior caso teorico 2^-8 = 3,91e-3);
`fix` — erro da media **6,24e-8** absoluto e **3,12e-5** relativo a `max|g|`
(limite teorico `nranks*step/2` = 6,25e-8) com `max|q| = 16000 <= 32767`, ou
seja a soma nunca estoura o int16. Por elemento, `fix` e' ~125x mais preciso que
`bf16` — e' o argumento para ele ser o "fp16" pratico. (Rodar antes de gastar um
job: `cd mpi && gfortran -O2 -o qprobe qprobe.f90 && ./qprobe`.)

**O que NAO muda:** nos dois modos todos os ranks terminam com o **mesmo**
buffer quantizado (o resultado vem do coletivo, nao do arredondamento local),
entao os ranks continuam bit-identicos entre si — o que muda e' a *trajetoria*
contra o run fp32. Isso e' o que `mpi/tests_quant.sh` mede, em 3 runs de mesma
config (`-n 2`, tau=1, mesmos dados nos 2 ranks, blas, 4 threads/rank, 40 passos,
checkpoints no passo 20 e 40):

1. **desvio numerico** dos pesos contra o run fp32: `max|dw|/max|w|` por tensor
   (mediana e pior), `||dw||/||w||` global e `max|dw|` absoluto, em dois pontos
   do run (para ver o crescimento com os passos);
2. **desvio de qualidade**: bpb no mesmo holdout de 300 linhas, medido pelo
   proprio app (`--nval 300`);
3. invariante entre ranks **com** quantizacao (`cmp` byte a byte, 90 `.npy`).

> **PENDENTE**: os numeros de (1) e (2) saem do job `mpiquant`
> (`logs/mpiquant_<id>.log`), que espera a fermi ficar livre (portao de
> concorrencia). A tabela de tau acima ja' esta' com o buffer de 19,02 MB.
>
> **Licoes dos dois primeiros jobs deste teste (114 e 116, ambos mortos):**
> (i) o 114 morreu mudo por teto de memoria abaixo da pegada fixa (4G < 4,4 GB);
> (ii) o 116 morreu no primeiro `import numpy` — o python do nix shell nao tem
> numpy, entao **qualquer script que toque checkpoint usa
> `/home/pauli/autoresearch/.venv-numpy/bin/python3`**; (iii) o 116 seguiu adiante
> depois disso e imprimiu "0 identicos, 1 diferentes" — glob nao expandido
> comparando diretorios que nao existiam. Agora todo run tem o rc checado com
> `check_rc` (log do run na tela + abort) e o checkpoint e' validado (90 `.npy`)
> antes de qualquer comparacao.

## 6. O que NAO foi verificado

1. **2 nos (a coisa que mais importa para tau=1).** Tudo aqui e' `mpirun -n 2`
   no mesmo no: o `Allreduce` sai por shm. O custo de rede, o
   `HYDRA_LAUNCH_AGENT`/ssh, e o fato de o binario nix depender de caminhos
   de `/nix/store` presentes nos dois nos **nao** foram testados.
2. **`sync delta` nao promedia os momentos do Adam.** So' os pesos: as 16
   `adam_*.npy` divergem entre ranks (esta' no log, 16 diferencas). Depois do
   primeiro merge cada rank continua com os seus momentos locais. A emulacao
   antiga tinha o mesmo comportamento; **este trabalho nao o corrigiu**, so' o
   tornou visivel.
3. **Energia**: o app MPI nao chama `fortran_energy` (o `train_run` chama, e o
   card `energy` sai nos checkpoints dele). Checks do `save_gpt_weights_st` com
   `energy=` ficam sem esse bloco no arquivo MPI.
4. **`--sync grad` com `H > 1`**: o codigo acumula os H micro-lotes locais e
   faz um sync + **um** update por bloco (e' DDP com gradient accumulation), o
   que muda o numero efetivo de updates de Adam em relacao a tau=1. Isso nao
   foi comparado com nenhum run registrado.
5. **Escala**: so' testado `-n 1` e `-n 2`. `-n 4/8` no mesmo no nao foi
   rodado (e o Allreduce de 38 MB x N ranks tem outro regime).
6. **Muon no MPI**: `--opt muon` compila e passa o `G` de geometria, mas nao
   foi rodado (o checkpoint de teste `/tmp/w_p2soup` nao tem momentos Muon).
7. **Tempo de parede das medidas**: o node esta' compartilhado com o job 103
   (8 CPUs). Os numeros de secao 5 sao limites superiores pessimistas.
8. **`--save_all`**: testado (rank 0 x rank 1 byte-identicos), mas nao com
   varios nos/DFSa.

## 7. Como rodar em 2 nos (e como rodar em 1 no sem derrubar ninguem)

**Orcamento de memoria (medido no incidente):** cada rank do `train_ddp` no
modelo d216/T=1024 fica em **760-845 MB** (pesos + Adam/Muon + cache de
ativacoes + temps + o npy do corpus). Duas consequencias praticas:

- `--mem` no job: **4 GB para `-n 2`**, 6 GB para `-n 4`, 10 GB para `-n 8`.
  Sem isso nao ha cgroup e qualquer excesso da maquina vira `global_oom` — que
  mata o processo MAIOR do no, tipicamente o `train_run` de um run longo.
- **nunca em paralelo com um run longo**: `jobs/fermi_mpiddp.sbatch` aborta se
  houver outro job RUNNING do usuario, e os scripts recusam rodar fora do Slurm.

Nada no codigo e' especifico de 1 no (so' `MPI_COMM_WORLD`), entao o passo para
dois nos e' o procedimento padrao do MPICH — **nao verificado**:

```bash
# (a) os dois nos precisam do MESMO binario e do mesmo fechamento de /nix/store
#     (o wrapper mpif90 do nix ja' embute os caminhos). Home compartilhado
#     resolve o binario; /nix/store tem de existir nos dois.
# (b) ssh sem senha entre os nos (o Hydra usa ssh para lancar os ranks)
# (c) processo 1 por no (cada rank le a sua fatia/linha; nada de dados no rank 0):
nix develop . --command nix shell nixpkgs#mpich.dev nixpkgs#mpich -c bash -c '
  mpirun -n 4 --hosts no1,no1,no2,no2 \
    mpi/build/mpif90_*/app/train_ddp \
    --weights DIR --rows ROWS.npy --out OUT --nsteps 2000 --lr 6e-4 \
    --ntrain 1024 --data rotate --sync grad --sync_every 1 \
    --save_every 200 --ckpt-format st
'
```
No SLURM com 2 nos, `srun --mpi=pmi2 -n 4 ...` ou `mpirun -n 4` com
`SLURM_NODELIST` (MPICH+Hydra entende `--launcher=slurm`), e `--data rotate`
garante lotes disjuntos sem nenhuma troca de dados entre ranks — os unicos bytes
que atravessam a rede sao os 38 MB do `Allreduce`.

## 7b. Cross-node (fermi + halfbeast): NAT, tuneis e a regra de seguranca

**Diagnostico.** A fermi esta' atras de NAT (IP privado `192.168.180.177`; o
halfbeast e' `143.107.180.212`). O `hydra_pmi_proxy` lancado no no remoto
precisa **conectar de volta** no launcher: o comando que o hydra monta (fonte do
MPICH 5.0.1, `lib/tools/bootstrap/external/external_common_launch.c`) e'

```
ssh -x [HYDRA_LAUNCHER_EXTRA_ARGS] <user@host> "<caminho do hydra_pmi_proxy>" --control-port <IP>:<PORTA> ... <proxy-id>
```

e o erro `Launch proxy failed / callback returned error status` e' exatamente
essa conexao de volta falhando. **Prova empirica**: com um listener em
`fermi:45999`, `ssh halfbeast 'echo > /dev/tcp/192.168.180.177/45999'` falha
(timeout, nada conecta). Nao era ssh quebrado, shell sujo nem caminho de proxy —
era rede.

**Solucao (tunelada, sem firewall nem root):**

| peca | o que faz |
|---|---|
| `mpi/crossnode_sshwrap.sh` | `-launcher-exec` do hydra: reescreve `--control-port <IP>:<P>` para `127.0.0.1:<P>` e acrescenta `-R <P>:127.0.0.1:<P>` ao ssh — o proxy "conecta em si mesmo" e o ssh leva ao launcher (canal de **controle**) |
| `mpi/crossnode_setup.sh` | um `ssh -N` com `-L`/`-R` para uma faixa FIXA (`9340-9359`) + `MPIR_CVAR_CH3_INTERFACE_HOSTNAME=127.0.0.1`, `MPIR_CVAR_CH3_PORT_RANGE=9340:9359`, `MPIR_CVAR_*_NETWORK_IFACE=lo` — resolve o canal de **dados** (ch3:tcp, portas dinamicas) |
| `mpi/allreduce_probe.f90` | sonda minima do caminho de dados (20x `Allreduce` de 1 MB); o `hostname` do mpirun **nao** exercita dados, so' controle/PMI |
| `-iface lo` no mpirun | o launcher escuta em `127.0.0.1` (senao o tunel apontaria para o vazio) |

O tunel foi **validado fora do mpirun**, nos dois sentidos: `halfbeast ->
127.0.0.1:9341` entrega em `fermi:127.0.0.1:9341` (controle) e `fermi ->
127.0.0.1:9340` entrega em `halfbeast:127.0.0.1:9340` (dados).

**REGRA DE SEGURANCA (requisito, nao preferencia).** Cross-node **sempre** com
`-envlist` explicito:

```
-envlist PATH,LD_LIBRARY_PATH,HOME,TMPDIR,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,<MPIR_CVAR_*>
```

Com o ambiente herdado (`-envall`, o default do hydra) **tudo** e' encaminhado
para o no remoto: num `-verbose` apareceram token de API e chave de terminal
indo para o halfbeast. Nenhum script deste repo usa `-envall`/`-genvall`; os
wrappers de cross-node fixam a lista acima.

**Estados de dados:** o binario, a sonda, `rows.npy`, `init/` (pesos + adam) e
`token_bytes.txt` ficam em `/tmp/ddp` nos DOIS nos (excecao consciente a regra de
`/tmp`: e' so' o launcher). O checkpoint do run sai no no do rank 0.

## 7c. Teto de escala: a decomposicao da pegada de memoria

Medido no job `memprobe` (110) com o MESMO `T=1024` e o mesmo corpus:

| modelo | parametros | RSS (plano, sem vazamento) |
|---|---|---|
| d96 (D=96, nh=6, nkv=2, nl=12, V=8192) | 2.752.512 | **4.403,6 MiB** |
| d216 (o do trabalho atual) | 9.510.912 | **5.323,6 MiB** |

Ajustando `RSS = F + c * params` a esses dois pontos: **F ≈ 3,93 GiB de custo FIXO**
(independente do modelo — contexto/framework) e **c ≈ 143 B/param** (137 B/param se
os numeros estiverem em MB em vez de MiB). Extrapolando:

- d360 (25M params): **≈ 7,8 GB** — cabe nos 15,4 GB da fermi;
- d768 (97M params): **≈ 18,1 GB** — **nao cabe** (e' por isso que distribuir, e
  nao so' aumentar a maquina, e' o caminho para a escala).

Leitura critica: `c ≈ 143 B/param` e' ~9x o nucleo fp32 do estado (params 4 +
gradiente 4 + `m` 4 + `v` 4 = **16 B/param**; se `m,v` fossem fp64 seriam 24 B/param,
ainda 6x menos). Somando os buffers que o codigo realmente aloca em d216/T=1024
(params 38 MB + Adam 76 MB + buffers Muon ~112 MB + cache de ativacoes ~150 MB +
temps ~250 MB ≈ **0,6 GB**), os 5,3 GB medidos **nao se explicam pelo modelo** — o
termo fixo de 3,93 GiB e a maior parte do coeficiente pedem atribuicao. Proximo
passo barato (para quem mede): `smaps_rollup` do processo separando **Anonymous vs
File-backed vs Shared**, com `--val_every` alto e um `rows.npy` minusculo, para
separar alocacao do trainer de *page cache* mapeado (o corpus de 63.889 linhas
mapeado e' ~262 MB; OpenBLAS com 8 threads aloca arenas por thread). Enquanto isso,
a regra pratica de memoria na fermi continua: **6G de teto por job** (`mpi/guard.sh`),
que e' o que impede um estouro de virar `global_oom` alheio.

## 8. Detalhes de implementacao que vale saber

- **Warmup identico ao `train_run`** (`lr * min(1, k/2)`, k relativo ao run).
  Sem isso o 1-rank e o K-rank nao comparariam (e' o que faz a prova da secao 4
  valer para o caso que gerou os runs registrados).
- **Fingerprint**: `fp` = XOR-rotate (13 bits) dos bits de todos os pesos, na
  ordem dos 8 arrays. Nao e' cripto: responde "os bits sao iguais?" dentro do
  log. A prova dura e' o `cmp` dos `.npy`.
- **Allreduce com 1 rank e' exato**: o app sempre passa pelo coletivo (o MPICH
  faz o memcpy local) e divide por `real(size)` = 1.0; `g/1.0 = g` em binario.
  E' isso que faz `-n 1` ser bit-identico ao `train_run` serial.
- **`--sync_log 1`**: imprime o tempo de cada `MPI_Allreduce` (foi assim que se
  separou custo do coletivo de contencao de CPU).
- **`die()` usa `MPI_Abort`**: um rank sozinho esperando num coletivo e' job
  pendurado; erro em qualquer rank mata todos.
- **Checkpoint sem fork**: `mkdir_p` (stdlib) e os diretorios de `step_*`
  pre-criados no setup, com os pools de OpenMP ainda frios.
- **`--mem` NAO EXISTE nesta fermi:** o `slurm.conf` reporta `RealMemory=1 MB`,
  entao `sbatch --mem=4G` e `--mem-per-cpu=512M` sao recusados ("Memory
  specification can not be satisfied"); os jobs do proprio supervisor rodam com
  `ReqTRES=mem=1M`. E' por isso que o OOM de 07:18 foi `global_oom` com
  `CONSTRAINT_NONE` — nao havia cgroup de memoria. O substituto e'
  `mpi/guard.sh` (escopo `systemd-run --user --scope -p MemoryMax=...`), que alem
  do teto **relata `memory.peak` e `memory.events` (max/oom_kill) e GRITA quando
  houve morte por memoria**.
- **O teto de 4G estava abaixo da pegada fixa do trainer.** O job 114 morreu MUDO
  na secao 1 (que roda o `train_run` PRE-refactor) porque a pegada medida e'
  **~4,4 GB de RSS plano** (job 110; nao e vazamento). Regra: **6G para qualquer
  teste que rode `train_run`/`train_ddp` com este modelo; 4G so' para MPI puro**
  (um `mpirun -n 2 train_ddp` mede ~1,7 GB). Use `mpi/guard.sh <CAP> <cmd>`.
- O binario MPI **nao** escreve `template.txt` (o `write_template_txt` e' do
  `train_run`); `arch.txt` e os pesos/Adam sao escritos em todos os formatos.

## 9. Estado de verificacao (resumo de uma linha cada)

- refactor puro: **provado bit a bit** contra o binario pre-refactor (90/90 `.npy`).
- invariante 2 ranks x 1 rank em tau=1: **bit a bit, zero diferencas**.
- ranks bit-identicos entre si (grad e delta, naive e blas): **sim**.
- data-parallel com lotes disjuntos (rotate, tau=1, blas, 4 threads/rank):
  **1640 -> 2759 tok/s (1,68x)**; e 2 ranks x 4 threads **batem** 1 x 8 threads
  (2460 tok/s) — secao 5a.
- `fortran-fpm test` do projeto principal: **verde** (`===0 failures ===` x2).
- 2 nos, energia, momentos no delta mode, `-n>2`: **nao verificado**.
- `--attn blas` e' o **default** desde 2026-09-18 (o naive ficou como opcao; a
  invariante foi provada com os dois kernels).
- `--sync-fp16 bf16|fix` corta o payload do coletivo para 19,02 MB (2x) mas e'
  **DEFAULT OFF**: `none` mantem a trajetoria bit-exata. Desvio (numerico e bpb)
  medido por `mpi/tests_quant.sh` no job `mpiquant` — secao 5c.
- operacao: `mpirun` so dentro do Slurm e nunca em paralelo com run longo (**regra
  dura**; incidente de 2026-09-18 07:18 documentado no topo). Como `--mem` e'
  recusado nesta fermi, o teto de memoria e' o escopo systemd do wrapper.
- buffer de um sync: **38,04 MB** = 9.510.912 floats (8 coletivos; d216) — secao 5b.
- o coletivo em `--sync grad` carrega **so' o gradiente**; os momentos nunca sao
  sincronizados e ficam identicos por construcao — confirmado pela comparacao
  `rank_0 x rank_1` que inclui os 16 `adam_*.npy` (90/90 byte-identicos).
