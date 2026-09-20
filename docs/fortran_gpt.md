# fortran_gpt — a biblioteca (o que tem dentro, como se usa)

Pilha de GPT em **Fortran puro, CPU-only, sem Python e sem torch** em nenhum
caminho de execução: forward/backward, laço de treino, KV/generação,
checkpoints safetensors e energia/CPU/IO **medidos pelo próprio processo**.
Este pacote é o `src/` (`src/fpm.toml`, fpm 0.13).

## Módulos (`src/lib/`)

| módulo | o que é |
|---|---|
| `fortran_arch.f90` | **a arquitetura em um lugar só** (D_MODEL, N_HEAD, N_KV, HD, N_LAYER, VV, TT, BOS) + `write_arch_txt`/`read_arch_txt`/`require_arch`/`check_shape` |
| `fortran_kinds.f90` | `wp = real32` (trocar para real64 num só lugar) |
| `fortran_blas.f90` | interface BLAS (atenção: o OpenBLAS do nix é **ILP64**) |
| `fortran_linear.f90` | `linear3d`, `wte_lookup` |
| `fortran_rmsnorm.f90`, `fortran_rope.f90` | RMSNorm, RoPE |
| `fortran_attn.f90` | atenção causal (naive + `sgemm`) |
| `fortran_gpt.f90` | forward |
| `fortran_backward.f90` | backward (gradientes) |
| `fortran_train.f90` | passo de treino (fwd+bwd+update) e val bpb |
| `fortran_adamw.f90`, `fortran_adam_state.f90` | AdamW e estado do otimizador (save/load) |
| `fortran_muon.f90` | Muon (ortogonalização Newton–Schulz) |
| `fortran_qkhop.f90` | QK-hop (experimentos de roteamento) |
| `fortran_kv.f90` | cache KV, `gpt_step` (decode incremental) |
| `fortran_spec.f90` | decode especulativo (drafter por lookup) |
| `fortran_recurrent.f90` | passes recorrentes (`--loops`) |
| `sample.f90` | amostragem (temperatura, top-k, penalidades) |
| `fortran_data.f90` | leitura/batching de linhas |
| `fortran_chat.f90` | template/formação de chat |
| `fortran_sys.f90` | `mkdir_p`, `dir_exists`, `exit` |
| `load_weights.f90` | checkpoints safetensors: pesos, estado do otimizador, metadata de arch, **card de energia** e linhagem |
| `tokenizer_tables.f90`, `tokenizer_encode.f90` | BPE em Fortran puro (tabelas exportadas por `scripts/export_tokenizer.py`) |
| `fortran_texture.f90` | **caracterização de textura em Fortran puro, sobre bytes**: alpha, byte_alto, palavra_plausivel, distinct-1/2/3, laço/rep + a ESCADA (chão de bytes → gerador usável → raciocinador) |

## Apps (`src/app/`)

| app | para que |
|---|---|
| `train_run.f90` | o treino de verdade: passos, eval, checkpoints (safetensors/.npy), trilha de energia |
| `train_1step.f90`, `train_loop.f90` | degraus menores de teste |
| `eval_bpb.f90`, `bpb_agg.f90` | bpb na holdout / agregação |
| `infer.f90`, `repl.f90`, `chat_text.f90` | geração: batch, REPL, chat |
| `merge_ckpt.f90` | merge (com/sem seleção) de checkpoints |
| `tier_probe.f90` | recomputar vs ler o KV — mede a fronteira de níveis de memória |
| `bench_attn.f90`, `bench_batch.f90`, `bench_gemm.f90`, `spec_bench.f90` | benchmarks de kernel |
| `tokdiff.f90` | diff de tokenizações |

## Testes (`src/test/`)

| teste | o que garante |
|---|---|
| `test_kernels.f90` | cada kernel contra referência inline, incluindo gradiente por diferenças finitas |
| `test_st_ckpt.f90` | round-trip safetensors + guardas de metadata (energia, linhagem, `self_measured`) |
| `test_energy_tier.f90` | invariantes do instrumento: page cache não toca o disco (`read_bytes`≈0 no quente), `FADV_DONTNEED` funciona (frio = bytes do disco), banda quente ≥ 0,5× fria, latência por IO ≥ 2× sequencial, contabilidade de tokens |

```bash
nix develop --command bash -c "cd src && fortran-fpm test"
nix develop --command bash -c "cd src && fortran-fpm test test_energy_tier"   # um só
```

## Build e gate

```bash
bin/fbuild              # build normal (-Wall -Wextra -fcheck=all -fbacktrace)
bin/fstrict             # GATE ESTRITO: -Werror, aplicado só no NOSSO código
bin/fbuild --werror     # delega para bin/fstrict
```

Shells do flake (nomes por **stack**; alias por **máquina**):

| shell | o que tem | quando usar |
|---|---|---|
| `.#cpu-only` | gfortran + fpm + openblas + python (sem runtime de GPU) | todo trabalho Fortran |
| `.#rocm` | cpu-only + rocmLibs + variáveis ROCm/HIP | só ferramenta de GPU (AMD) |
| `.#default` | = `.#cpu-only` | entrar sem argumento não deve baixar GB de GPU |
| `.#fermi` | = `.#rocm` | na fermi (GPU AMD gfx1100) |
| `.#halfbeast` | = `.#cpu-only` | na halfbeast (Intel, sem GPU útil) |
| `.#inference` | = `.#cpu-only` | nome antigo, mantido como alias |

Os dois stacks usam o **mesmo nixpkgs rev** (flake.lock), então gfortran e
OpenBLAS são as mesmas derivações em qualquer máquina — é isso que torna bpb e
tok/s comparáveis entre fermi e halfbeast. O shell traz também o alias `fpm`
(apontando para `fortran-fpm`).

`bin/fstrict` compila tudo com flags default e depois **recompila só os nossos
fontes** com `-Werror`. Motivo: o `--flag` do fpm também vai para as dependências
(stdlib emite `compare-reals` 104×, `conversion` 93×, `unused-dummy-argument`
20×); um `-Werror` global exigiria uma lista de exceções que enfraqueceria o
gate para o código que importa.

Use `--build-dir /tmp/...` para verificar sem tocar uma árvore de build que jobs
em fila/execução vão usar (os binários são resolvidos no momento da execução).

Dependências: `stdlib` (registry), `openmp`, `safetensors` (git, tag `v0.1.2`),
`M_CLI2` (git, **commit** `0704ed3`: a tag V3.2.0 difere em 1109 linhas do que
estava vendado), `fortran_energy` (path — vira git quando for publicado).

## Textura: o checkpoint se descrevendo

`fortran_texture_mod` mede a textura **sobre bytes** (caractere = byte via
`iachar`; decodificar antes de medir introduz juízo de valor e muda o painel) e
`app/texture_scan.f90` aplica a um arquivo:

```bash
bin/fbuild && ./build/*/app/texture_scan AMOSTRA.txt --json T --key texture
```

A régua é **calibrada nos controles deste repo** (ver `test_texture.f90`, que
testa as invariantes): prosa real `palavra_plausivel` ~0,95 e `byte_alto` ~0,005;
modelo 3M treinado (2,33 bpb) **0,20 e 0,146 → degrau 1 (chão de bytes)**.
`distinct_2` é invertido (salada tem mais, porque não tem a redundância da
língua), e `maior_laco`/`rep_frac` só contam n-gramas não-brancos (senão linha de
separador de markdown vira "laço").

A saída JSON é a mesma linha que deve ir para o `__metadata__` do checkpoint
(`texture.*`), para o card dizer **de antemão** o que o modelo é. Integração
pendente: o gerador hoje vive dentro de `app/repl.f90` (programa), então falta
extrair a geração para um módulo e ter um app que amostra o próprio checkpoint e
grava a textura no card.

## Identidade da arquitetura (P1/P2 feitos; P3/P4 pendentes)

**O problema (medido):** a verdade sobre a arch morava em **três lugares** — os
`parameter` de `fortran_arch.f90` (o que o binário executa), o `__metadata__` do
safetensors (já escrito: `arch.d_model` …) e o sidecar `arch.txt` — e o vínculo
binário↔checkpoint era um **nome de pasta** (`arch_d96_h6_kv2_l12_v8192_c1024`,
schema repetido em `fedavg_rounds.py`, `compose_grid.py` e `set_arch.sh`). Foi
assim que a árvore ficou com a fonte em **d216** enquanto os experimentos em voo
são **d96**, sem nada estrutural impedir.

**A identidade (agora):** derivada dos próprios parâmetros, sem nada a manter em
sincronia.

| peça | o que faz |
|---|---|
| `fortran_arch_mod::arch_canonical()` | string canônica (ordem fixa, sem espaço) |
| `fortran_arch_mod::arch_id()` | 16 dígitos hex: **identidade** para seleção/checagem |
| `app/arch_id.f90` | `arch_id` = arch compilada; `arch_id --from DIR` = a do checkpoint |
| `load_weights_mod::read_arch_any()` | leitor **único**: `__metadata__` primeiro, `arch.txt` como fallback |
| `check_arch_selfconsistent()` | o checkpoint tem de ser consistente **consigo** (campos ↔ canônica ↔ id) |
| `scripts/arch.py` | a mesma definição em Python (+ CLI) |
| `bin/arch_check.sh` | **teste de concordância Fortran↔Python** nos dois portadores |
| `test_arch_id.f90` | determinismo, sensibilidade campo a campo, não-colisão, forma |

O `arch.id` **não é criptográfico**: é rotação+xor sobre os bytes da canônica,
escolhido porque (a) não depende de overflow de inteiro assinado (que a norma não
define e o compilador pode explorar) e (b) tem implementação idêntica nos dois
lados — o que é o que permite o teste de concordância. Ele só precisa não
colidir entre configurações, e isso é testado.

Exemplo (d96 dos experimentos em voo): `id = ca674ae5302a6cc9`. O binário da
árvore hoje (d216) tem `bec469fb7c561d2c` — **a divergência virou duas
identidades explícitas** em vez de um nome de pasta.

**Pendente:** P3 = selecionar binário por identidade (`build/arch_<id>`,
`pick()`/`compose_grid.py` deixam de duplicar o schema do nome) e o driver
conferir `arch_id`; P4 = aposentar o `arch.txt` (primeiro derivado, depois fora),
com entrada no HEP.

## Retrato da arquitetura

Um comando imprime como a arquitetura está **agora** — diagrama do bloco com as
dimensões, tabela de tensores (shape e params), os derivados que decidem custo
e a **identidade**:

```bash
scripts/arch_view.py /tmp/mix/init3m --repo-default --svg /tmp/arch.svg
```

Ele lê a arch do **checkpoint** (metadata primeiro, `arch.txt` depois) e, com
`--repo-default`, compara com os `#define ARCH_*` do repo — imprimindo
`>>> DIVERGENCIA` quando um build novo sairia para outra arquitetura. Foi
precisamente essa a confusão d216/d96: agora ela aparece em uma linha em vez de
esperar alguém conferir campo a campo.

Derivados que saem junto (e que decidem decisões): params, FLOPs/token (fwd e
fwd+bwd), atenção/token, **KV B/token**, **ativação B/token** (o `allocate(C%…)`
do `fortran_train.f90`), estado do otimizador (m,v) e o que um lote de T=1024
ocupa. Com uma ressalva medida embutida no texto: **FLOPs ≠ custo na atenção** —
no `tier_probe`, atenção custou 225 µs/token contra 83 µs/token das projeções
com ~0,6× dos FLOPs (ela é dominada pelo tráfego da matriz T×T).

## Convenções que evitam bug

1. **A arquitetura é uma só**: shapes em `fortran_arch_mod`, derivados, nunca
   digitados. Trocar de tamanho é `scripts/set_arch.sh` (reescreve o módulo) —
   a mudança aparece no diff.
2. **Shape errado falha alto**: `require_arch`/`check_shape` existem por causa de
   um caso real de lixo silencioso com binário/checkpoint incompatíveis.
3. **Checkpoint carrega mais que pesos**: `arch.*`, `energy.J`,
   `energy.self_measured` (a medida é do processo, não rateada por job),
   `lineage.parent`.
4. **Energia por fase**: `call energy_mark('nome', tokens=n)` no caminho quente;
   a trilha vai para `energy_trace.csv` e o card do checkpoint leva o delta.
5. `wp` é `real32`; BLAS é ILP64 (ver `fortran_blas.f90` antes de mexer).

## Onde continuar

- `docs/tier_probe.md` — recompute vs leitura de KV, medido (`hyp_2ac980`).
- `docs/sync_composition.md` — o que custa sincronizar réplicas de treino.
- `docs/mpi_ddp.md`, `docs/halfbeast.md` — multi-máquina e fila.
- Lacunas conhecidas: `fortran_energy` ainda é dependência por caminho (a
  publicação do `energy-fortran` está em andamento) e não há CI neste repositório
  (ele depende das máquinas do lab); `safetensors-fortran` e `energy-fortran` têm
  CI próprio.
