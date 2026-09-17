# Backend safetensors no trainer (fase 1) — relatório

Escopo: `/home/pauli/autoresearch` (o trainer), **não** a biblioteca.
Biblioteca: `github.com/14NGiestas/safetensors-fortran`, pinada em `src/fpm.toml`
como dependência git, agora na **tag v0.1.1** (ver §5: achei e consertei um bug
upstream).

Regras respeitadas: nada de treino longo (só passos de 1 step em scratch, 8 s cada),
`scripts/set_arch.sh` não foi rodado, arch default (d216) intocada, nada escrito sob
`/tmp`, `fortran_train.f90` / `fortran_qkhop*` / `jobs/*.sbatch` não foram tocados,
nenhum commit no repo do trainer.

---

## 1. Entregáveis

| # | Entregável | Estado | Onde |
|---|---|---|---|
| 1 | lib no build | **dependência git, tag v0.1.1** (não vendorizado) | `src/fpm.toml` |
| 2 | `--ckpt-format npy\|st\|both` (default npy) | pronto, comportamento atual intacto | `src/app/train_run.f90` |
| 3 | leitura safetensors com auto-detecção | pronto (npy **ou** `model.safetensors`, sem flag) | `src/lib/load_weights.f90` |
| 4 | testes em `src/test` | `test_st_ckpt.f90` (byte a byte + Python + verificação de corrupção) | `src/test/test_st_ckpt.f90`, `scripts/st_read.py` |
| 5 | relatório | este arquivo | `docs/safetensors_backend.md` |

### 1.1 Dependência (não vendorizada)

```toml
safetensors = { git = "https://github.com/14NGiestas/safetensors-fortran", tag = "v0.1.1" }
```

O fpm clona a dep em `<build-dir>/dependencies/safetensors` (não há cache global:
`~/.local/share/fpm` não existe nesta caixa). Consequência medida, não suposta:

| cenário | resultado |
|---|---|
| build offline com `build/dependencies/` já populado | `Project is up to date` — **OK** |
| build offline com build dir **novo** (`rm -rf build`) | `Error while fetching git repository for remote dependency` — **falha** |

Ou seja: depois do primeiro fetch o build é offline; um build dir novo precisa de
rede **uma vez**. Não vendorizei (decisão sua); se em algum momento precisar de
build offline garantido em dir novo, o caminho é copiar `src/safetensors.f90` e
`src/safetensors_json.f90` da dep para `src/lib/` e tirar a linha do `fpm.toml`
(módulo `safetensors` definido duas vezes = erro de link, então os dois não podem
coexistir).

Efeito colateral a saber: cada `--build-dir arch_*` tem o **seu** clone. As pastas
`src/build/arch_d96_*` usadas pelo job 86 **não** têm a dep; os binários já
compilados seguem rodando (não os toquei), mas um rebuild futuro daquela pasta
precisa de rede uma vez para trazer a dependência.

---

## 2. Comandos e resultados (evidência)

### 2.1 Build e suíte completa (o que o critério de pronto pede)

```bash
cd /home/pauli/autoresearch
nix develop --command bash -c "cd src && fortran-fpm build"   # 8 s, Project compiled successfully
nix develop --command bash -c "cd src && fortran-fpm test"    # 30 s, exit 0
```

`fortran-fpm test` roda os **dois** programas de teste e termina verde:

```
===0 failures ===                       <- test_kernels (10/10 kernels, inalterado)
== safetensors backend: round-trip byte a byte ==
  ok    wte: payload BYTE A BYTE igual ao .npy
  ... (14 tensores, 28 checagens de bytes)
  ok    verify PEGA payload truncado: .../truncated/model.safetensors
        (tensors need 9728 bytes but the file only has 9724 after the header
         (truncated payload))
  ok    arch.d_model errado aborta com exit=1 (obtido 1)
  ok    python: payloads .st == .npy, card e arch.* validos
===0 failures ===
```

`src/test/test_st_ckpt.f90` cobre: escrita nos dois formatos, comparação **byte a
byte** dos 14 tensores contra os `.npy` (com NaN/+Inf/−Inf/−0.0/denormal no
payload), leitura auto-detectada, diretório com **só** `model.safetensors`, metadata
com acento + chave vazia, `verify_ckpt_dir` nos dois modos, payload truncado
(o acidente de disco cheio que motivou o verify), aborto por `arch.*` divergente
(subprocesso, `exit=1`) e a leitura por Python.

### 2.2 `--ckpt-format` na CLI

```bash
train_run --help | grep -A8 ckpt-format      # opção documentada
train_run ... --ckpt-format npyz             # exit 2, "require --ckpt-format npy|st|both"
train_run ... --ckpt-format st --anneal 1    # aceito (parava em "unknown keyword" antes, §5.2)
```

### 2.3 Passo real (1 step, scratch) nos três modos — checkpoint de verdade

Pesos de partida: `scripts/make_init.py --d 216 ... --seed 7` (37 MB, fora de `/tmp`);
rows sintético de 4 linhas; `--bytes ~/.cache/autoresearch/tok_tables/token_bytes.txt`.

```bash
TR=$(ls src/build/*/app/train_run | head -1)
$TR --weights dev/st/smoke/winit --rows dev/st/smoke/rows.txt \
    --out dev/st/smoke/out_both --nsteps 1 --ntrain 3 --nval 1 --trn_probe 1 \
    --val_every 1 --save_every 1 --attn blas --ckpt-format both --bytes $TB
```

| modo | tempo | conteúdo do `step_1/` | conclusão |
|---|---|---|---|
| `both` | 8,5 s | 92 arquivos + `model.safetensors` (38.049.584 B) | as duas representações do mesmo passo |
| `st` | 7,9 s | `model.safetensors` + 16 `adam_*.npy` + `arch.txt` + `template.txt`, **0** `transformer_*.npy` | só o peso mudou de representação |
| `npy` (default) | 8,0 s | 92 arquivos, **0** `model.safetensors` | comportamento atual intacto |

Os `.npy` do modo default são **byte-idênticos** aos `.npy` do modo `both`
(`cmp` de `transformer_wte_weight.npy`) — `both` não perturba o caminho legado.
E os dois passos independentes (`both` e `st`, mesma init/rows/lr) produziram
`model.safetensors` **byte-idênticos**: o passo é reproduzível bit a bit nesta caixa.

### 2.4 Paridade dos payloads num checkpoint real (Python, independente)

```bash
.venv-numpy/bin/python3 scripts/st_read.py \
    --ckpt-dir dev/st/smoke/out_both/step_1 --compare-npy
```

```
st      : .../model.safetensors (38049584 bytes, 74 tensores, 13 chaves de metadata)
card    : steps=1 lr=0.000150000007 tokens=1024 rows_file=dev/st/smoke/rows.txt metrics={}
arch    : d=216 heads=6 kv=2 layers=12 vocab=8192 ctx=1024 head_dim=36
tensor  : wte        F32  [1769472]  7077888  fnv1a64=b40fe7ee73e8f193  .npy OK
... (74 tensores, todos ".npy OK")
st_read: OK
```

`scripts/st_read.py` não usa numpy nem o pacote oficial: lê com o oráculo
pure-Python **da própria biblioteca** (`reference_writer.py`, encontrado no clone
local ou em `src/build/*/dependencies/safetensors/tools/`), que valida o header
inteiro (JSON, dtypes, offsets, cobertura) e compara os bytes com os `.npy`.

### 2.5 Deliverable 3: app real lendo um diretório só-safetensors

```bash
$EVAL --weights dev/st/smoke/out_st/step_1 --rows dev/st/smoke/rows.txt --attn blas --batch 1
  # stderr: weights: .../out_st/step_1/model.safetensors (safetensors, canonical names)
  # exit 0, 4 linhas de NLL
printf '%s' "Alan Turing theorized that computers" | \
  $CHAT --tables ~/.cache/autoresearch/tok_tables --weights dev/st/smoke/out_st/step_1 --n 4
  # exit 0, 51 bytes gerados
```

E `cmp` das saídas do `eval_bpb`: **NLL(st-only) == NLL(npy)**, bit a bit
(as duas árvores de diretório têm pesos byte-idênticos). Os apps não precisam
saber qual formato está lá — a escolha é feita dentro de `load_gpt_weights`, pelo
`inquire` do `model.safetensors`.

Detalhe de implementação: o aviso de qual caminho foi lido vai para o **stderr**
(unit 0). O stdout do `eval_bpb` é parseado por `scripts/eval_driver.py` (linhas de
NLL); um "loading..." ali quebraria o driver em silêncio.

### 2.6 `arch.*` do metadata é conferido (não é enfeite)

Um diretório só-safetensors não tem `arch.txt`, então `require_arch` não teria o que
checar. A checagem foi para o caminho de leitura: `load_gpt_weights_st` compara
`arch.d_model/n_head/n_kv/n_layer/vocab/head_dim` do `__metadata__` com o que o
binário espera e **aborta** em divergência (mesma política do `arch.txt`, mensagem
`FATAL: architecture mismatch (arch.d_model): this binary expects ... | file declares`).
Chave ausente é ignorada (arquivo de outra ferramenta). Os modos `st`/`both` também
continuam escrevendo `arch.txt`, então `require_arch` segue funcionando neles.

---

## 3. O que mudou (arquivos)

| arquivo | mudança |
|---|---|
| `src/fpm.toml` | +1 linha: dependência git da lib (tag v0.1.1) |
| `src/lib/load_weights.f90` | `save_gpt_weights_st` (escrita), `load_gpt_weights_st` + helpers, `load_gpt_weights` com auto-detecção, `check_declared_arch`/`cmp_meta`, `verify_ckpt_dir` com `st_only=` |
| `src/app/train_run.f90` | `--ckpt-format` (validação + 4 pontos de escrita + verify ciente do formato); `--anneal 0` no spec do M_CLI2 (§5.2) |
| `src/test/test_st_ckpt.f90` | novo (teste byte a byte + integração + subprocesso de arch) |
| `scripts/st_read.py` | novo (leitor/validador independente, não usa numpy) |
| `docs/safetensors_backend.md` | este relatório |

Nomes canônicos: `wte`, `lm`, `l{ll}.q/k/v/p/fc/p2` (1-D float32 flat, iguais aos
`.npy`). `__metadata__`: `format_version`, `producer`, `arch.source`,
`arch.d_model/n_head/n_kv/n_layer/vocab/ctx/bos/head_dim`, `n_tensors` e `card`
(JSON com `steps`, `lr`, `tokens`, `rows_file` e `metrics` **vazio** — quem treina
preenche depois; número inventado aqui viraria verdade no experimento).

---

## 4. O que NÃO foi verificado

- **Checkpoint real de experimento (d96) convertido/aberto**: os testes com pesos
  reais que rodei são d216 (arch default). Converter um checkpoint d96 exigiria o
  binário d96 (`arch_d96_*`, em uso pelo job 86) — não toquei.
- **`merge_ckpt` / `scripts/merge_checkpoints.py`** com diretórios safetensors: não
  mudei esses caminhos; a auto-detecção vale para eles (leem via
  `load_gpt_weights`), mas **não** foi exercitada em merge.
- **Grids `compose`/`fedavg`** (`--anneal`): não rodei nenhum. O job 86 está com
  binários antigos (ainda com o bug do §5.2) — ver a nota lá.
- **`t0 > 1`** (retomada) com `--ckpt-format st`: o `card.steps` usa `tstep` absoluto
  (ok), mas só testei `t0=1`.
- **Arquivos > 2 GB** e **`wp = real64`**: caminhos não exercitados (o loader fixa
  `real32` de propósito, `st_load` converte).
- **CI do repo da lib** com a tag v0.1.1: o workflow foi atualizado (flag nova) mas
  não roda daqui.
- **BF16/FP16** nos checkpoints: fora de escopo (a lib lê cru via `get_raw`).
- **`verify_ckpt_dir` em modo `st` com disco cheio de verdade**: testei com o
  arquivo truncado de 4 bytes a menos (o sintoma), não enchendo um disco.

---

## 5. Dois bugs encontrados no caminho (consertados)

### 5.1 Biblioteca upstream: tabela de dígitos hex truncada (v0.1.1)

`src/safetensors_json.f90` declarava

```fortran
character(len=4), parameter :: hexd = '0123456789abcdef'
```

O inicializador de 16 chars era **silenciosamente truncado para `'0123'`**, e todo
escape `\u00xx` lia fora do fim da constante: `0x1F` saía como `\u001A` (dígito
lixo + leitura fora dos limites + valor que não sobrevive ao próprio round-trip).

Por que passou: **nem `-Wall` nem `-Wextra` avisam disso** — precisa de
`-Wcharacter-truncation` (que a suíte da lib não usava). E o fixture de escapes
cobria só `\t \n \r \" \\`, DEL e UTF-8: nenhum caminho `\u00xx` era exercitado.

Conserto (commit `d50b39a`, tag **v0.1.1**, `main` e tag **empurrados** para
`origin`):

- `character(len=16)` (o tamanho real da tabela);
- o fixture de escapes passou a carregar **todos** os bytes de controle
  `0x01..0x1F` (os cinco de escape curto + os `\u00xx`), continuando byte-idêntico
  ao que o `serde_json` oficial escreve;
- teste de round-trip conferindo cada um dos 31 bytes;
- `-Wcharacter-truncation` entrou nas flags do CI e na seção de build do README;
- prova de regressão: reintroduzindo o `len=4`, a suíte vai a **PASS 105 / FAIL 3**;
  com o conserto, **PASS 108 / FAIL 0** (e `cmp` idêntico nos dois fixtures).

O trainer já consome a v0.1.1 (`src/build/cache.toml`: `rev = d50b39a...`) e a prova
ponta-a-ponta pelo caminho da dependência dá `\u001f` correto, com leitura de volta
do byte 31.

### 5.2 Trainer: `train_run` estava morto para QUALQUER invocação (`--anneal`)

Bug **pré-existente**, não introduzido por mim, e que você deve querer saber antes
do resto: `--anneal` era lido com `sget('anneal')` mas **nunca entrou no spec do
`set_args`**. Com o M_CLI2 vendorizado isso dá:

```
sem --anneal :  *get_anyarray_c* unknown keyword anneal          (stop 8)
com --anneal :  UNKNOWN LONG KEYWORD: --anneal  + tabela de uso   (stop)
```

Reproduzido com o binário **antigo** (d216, compilado antes de eu tocar em nada):

```
$ src/build/arch_d216_*/gfortran_CC0F834C608B9708/app/train_run \
    --weights /nonexistent --rows /nonexistent --out /tmp/x --nsteps 1
*get_anyarray_c* unknown keyword anneal
$ ... --anneal 1
UNKNOWN LONG KEYWORD: --anneal
```

Ou seja: `compose_grid.py` (job 86, família `anneal`) e `fedavg_rounds.py` (jobs
87–89) passam `--anneal 1` e não podem funcionar; e qualquer job que **não** passe
o flag morre no mesmo lugar. Conserto: `--anneal 0` no spec (uma linha), que
preserva o default documentado (LR constante) e passa a aceitar `--anneal 1` como
cosine. Verificado: os dois modos agora passam da validação de argumentos e param
só onde deviam (no load de pesos, porque apontei para um dir inexistente).

**Ação necessária do seu lado**: os binários `arch_d96_*` usados pelo job 86 são
anteriores a esse conserto. Enquanto não houver rebuild daquela pasta (não fiz, por
instrução), os passos de treino daquela grid continuam morrendo com a mensagem
acima.

---

## 6. Próximos passos sugeridos

1. Rebuild da pasta d96 (com rede, uma vez, para a dependência) e um
   `--ckpt-format both` num checkpoint real de experimento — fecha o único buraco
   relevante da fase 1 (checkpoint de verdade, não sintético).
2. Fase 2 do backend: `load_gpt_weights` pode passar a **preferir** safetensors e
   tratar o `.npy` como legado (hoje a detecção é só por presença, sem preferência),
   e o `merge_ckpt` pode escrever `both` para que a média de checkpoints já saia nos
   dois formatos.
3. Levar `arch.*`/`card` para o lado Python (`scripts/eval_driver.py`,
   `scripts/compose_grid.py`, `fedavg_rounds.py`): ler bpb/linhagem do
   `__metadata__` em vez de inferir do nome do diretório, que é o motivo original de
   o card existir.
