# tier_probe — recomputar vs guardar/ler (inferência e treino)

Instrumento: `src/app/tier_probe.f90` (dois modos, mesmo instrumento) sobre
`src/lib/fortran_probe.f90` (probe da máquina, **sem nada de GPT**) e
`energy-fortran` (J/tempo/CPU/IO **no processo**). Shapes do modelo vêm do
`arch.txt` do checkpoint (`--ckpt`), não do `fortran_arch_mod`.

A tese: onde fica a fronteira "recomputar vs ler" **não é uma propriedade do
algoritmo, é razão compute:IO da máquina**. O mesmo binário, em duas máquinas,
dá fronteiras que diferem ~380×.

## Modo `infer` — janela de KV no decode

| nível / operação (1 token = 3 KB no d96) | fermi (NVMe, **ociosa**, medição 139) | halfbeast (HDD) |
|---|---|---|
| RAM sequencial (page cache) | **0,11–0,12 µs** (25–29 GB/s) | 0,375 µs |
| RAM, 1 IO/token (aleatório) | **7,5–8,1 µs** | 15,6 µs |
| disco sequencial | **1,2 µs** (2,5 GB/s) | 26,9 µs |
| disco, 1 IO/token | **102–107 µs** | **9.732 µs** |
| recomputar a janela (proj+atenção) | 81 / 128 / **257** µs/token (T=32/128/512) | 309 µs/token (T=512) |
| **N\* (empata com ler token-a-token, frio)** | **66–88 tokens** | **18.984 tokens** |

(Os números da fermi na tabela são da medição limpa — job 139, máquina ociosa,
5 repetições por célula. Sob contenção a banda quente caía para 9–22 GB/s e o
N\* ia para ~50: medir máquina ocupada dá lixo, e o `probe_trust` reprova.)

- Contra **RAM**, ler ganha sempre (10–2000×).
- Contra **disco token-a-token**, recomputar vence para janelas ≲ N\*; acima disso,
  ler ganha (na halfbeast, recomputar 512 tokens = 158 ms contra 4,98 s lendo).
- **Lote é a variável de controle**: lendo a janela em **um** IO, ler ganha por
  **17,8×** (T=32), **63,5×** (T=128) na fermi e ~4–7× na halfbeast.
- A **atenção** é O(N²) na janela (O(N) por token): recomputar piora conforme a
  janela cresce.

A leitura do vídeo/paper de fronteira — "delete a memória local e recompute 128
tokens" — é uma afirmação sobre a razão FLOPs/IO *do hardware deles*: uma GPU tem
~100× os FLOPs de um core de CPU com a **mesma latência de NVMe**, o que empurra
N\* de ~50 para a casa dos milhares e deixa 128 tokens trivialmente do lado do
recompute.

## Modo `train` — ativações e estado do otimizador

O footprint de ativação **não é chutado**: vem do `allocate(C%…)` do
`fortran_train.f90` — `C%e, C%xa, C%q, C%ao, C%e1, C%qr = 6·d_model`, `C%f = 4·d_model`
(`dff = 4·DD`), `C%k, C%v, C%kr = 3·d_kv`. Nosso d96: **1.056 floats por camada por
token (4.224 B)** → **50.688 B/token** no modelo todo (17× o KV), e um lote de 512
tokens guarda **25,95 MB**.

| medido (d96, 2,75M params) | valor |
|---|---|
| recomputar 1 camada/token | 7,37 µs |
| ler a ativação dela (seq) | RAM 0,31 µs · disco 1,93 µs |
| **T\*** (leitura dispersa, 1 IO/token) | RAM 1,6 · disco 32,4 tokens |
| estado do otimizador (m,v = 2P) | 22,02 MB |
| offload de m,v 1×/passo | RAM 1,60 ms · disco 10,07 ms (forward do lote = 45,25 ms) |

Vereditos que saem direto dos números:

1. **Guardar ativação ganha de recomputar em todos os níveis** — o oposto da
   inferência: aqui a memória é barata e o recompute caro (num core de CPU).
   Ou seja: **não vale fazer activation checkpointing neste modelo/escala**; o
   regime em que ele paga é o de ativações que não cabem na memória rápida — não
   é o nosso.
2. **Offload de m,v para disco custa ~22% de um passo** → manter em RAM, ou
   encolher para bf16 (metade dos bytes), **ou não carregar entre syncs** — que é
   exatamente a decisão que o braço C do experimento federado tomou pelo lado da
   qualidade (−0,196 bpb). Dois caminhos independentes apontando a mesma escolha.

## Sensores de energia: o limite honesto (corrigido)

- **fermi**: `/sys/class/powercap/intel-rapl:0/energy_uj` é `-r-------- root root`
  — sem privilégio **não se mede energia de CPU**. O único sensor legível é
  `hwmon7 = amdgpu` (~50,14 W ociosos), e o app agora **avisa em voz alta** que
  esse J é da GPU. Prova aritmética do que aconteceu: 56,8 J em 1,1008 s ≈ 51,6 W
  = a potência ociosa da GPU.
- **halfbeast**: `energy_uj` é `-r--r--r--` (i9-7900X, contador real) → é lá que
  energia de CPU funciona. Números de energia **da CPU**: lendo do disco
  aleatoriamente custa **1,36e-1 J/token** contra **1,19e-2 J/token** recomputando
  → ler custa **11,5× mais energia** no regime disperso.
- Corolário: `energy.J` de checkpoints treinados **na fermi** mede GPU ociosa, não
  o treino. Energia só a partir da halfbeast (ou de um contador legível).

## Regras de honestidade embutidas no probe

- **frio verificado**: `read_bytes` do `/proc/self/io` tem que ficar ~0 no quente e
  ≈ bytes no frio (`probe_trust`).
- **mediana de N repetições + spread**, passada de aquecimento, `fsync` + 1 s de
  quiescência depois de escrever (sem isso o writeback contamina e o spread passa
  de 1000%).
- **recusa a certificar** célula instável (ex.: com load 8,8 na fermi o rep1 deu
  1,1 s contra 5 ms dos outros — desligamento de CPU); `PROBE_VERBOSE=1` imprime o
  wall de cada repetição.
- **contexto** registrado por célula: `cores_busy`, `cpu_pct`, `rd_MB`, `reps`.
- `--json` anexa uma linha por célula (o número pode ser citado por um run).

## Refs

- `hep/` — hipóteses `hyp_2ac980` (fronteira compute:IO) e `hyp_8f9016`
  (saturação em K).
- `docs/fortran_gpt.md` — a biblioteca; `docs/sync_composition.md` — composição.
