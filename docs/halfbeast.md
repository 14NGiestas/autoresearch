# halfbeast — fila remota para trabalho CPU-bound

Máquina compartilhada, sem GPU. Usada para: builds fpm, testes, treinos CPU/OpenBLAS.
Despacho por `ssh halfbeast` + Slurm (`srun`/`sbatch`). **Sem sudo, e não é preciso.**

## Fatos (verificados)

| item | valor |
|---|---|
| host | `halfbeast` (ssh como usuário comum, sem senha) |
| CPU | 20 threads = **10 núcleos físicos** (i9-7900X) |
| RAM | 125 GB, e **configurada no Slurm** (120428 MB) -> `--mem` funciona aqui |
| partição | `compute` (default, 1 nó, 20 CPUs) |
| nix | **2.6.0** — velho demais para o nosso nixpkgs pinado |
| GPU | nenhuma (`/dev/kfd` ausente); GPU fica no fermi |
| outros usuários | sim — ex.: job 6010 de `iangiestas` |

### Consequência do nix 2.6.0
Não dá para `nix develop` o nosso flake lá. Duas saídas:
1. **Ship o binário** (nossa rota): `bin/sync_eval.sh` manda o binário + closure de
   `.so` + wrapper `ld.so` + tabelas para `~/autoresearch-eval/`, e o job chama
   `portable/run_train_run.sh` / `run_eval_bpb.sh` / `run_chat.sh`.
2. Se precisar de ISA específica, compilar *na* ferm (aí o binário é para lá, não
   para cá) — de preferência opção 1.
Flakes, se um dia rodar nix lá: `--extra-experimental-features 'nix-command flakes'`.

## Regras de boa vizinhança (embutidas no `bin/hb.sh`)

1. **Checar antes**: `bin/hb.sh status` mostra carga, fila e memória livre. Se
   `load` estiver alto ou houver fila de outro usuário, o despacho recusa (a não
   ser com `--force`).
2. **Nunca os 20 threads**: HT atrapalha nos dois boxes (medido). Job "cheio" é
   `-c 10`; com outros rodando, `-c 8` ou menos.
3. **Off-peak**: trabalho longo (horas) prefere noite/madrugada.
4. **`nice` quando não for Slurm**: coisa curta e não-Slurm roda com `nice -n 19`
   (só come ciclos ociosos).
5. **Não monopolizar**: os 20 CPUs são de todos; quem tiver fila manda.

## Comandos

```bash
bin/hb.sh status                      # carga, fila, memória, partição
bin/hb.sh run 'comando'               # srun -c 8 na partição compute (com pré-check)
bin/hb.sh batch jobs/algum.sbatch     # sbatch (o script declara -c/-t)
bin/hb.sh sync                        # envia binário+closure+tabelas para lá
bin/hb.sh logs [N]                    # últimos logs dos jobs
```

## Traps que já nos custaram tempo

- **Pasta de build ambígua**: rodar binário de outra arquitetura. Aqui é pior,
  porque o binário viaja por rsync. Sempre nomear o bundle pela arquitetura
  (`arch_d96_...`) e conferir com `arch.txt` no checkpoint (`require_arch`).
- **`pkill -f <padrão>`**: se o padrão aparecer na sua própria linha de comando,
  você mata seu shell (aconteceu 3x). Use `pgrep -f '[p]adrao'` e mate por PID.
- **`du`/mtime de saída do wikiextractor**: ele bufferiza; "arquivo parado" não
  significa processo morto. Confira com `ps`, não pelo mtime.

## Nix na halfbeast (instalado 11/set, 22h)

Instalado o **nix oficial multi-user** (daemon), substituindo o `nix-bin 2.6.0` do apt
que não abria o nosso flake (precisa ≥2.18). Versão: **2.35.2**.

```bash
# o que foi feito (via tmux 0:4, sudo digitado pelo dono):
sudo apt-get remove --purge -y nix-bin nix-setup-systemd   # pacote da distro
sudo userdel <cada nixbld*>; sudo groupdel nixbld          # restos do apt (UID 998)
sh <(curl -L https://nixos.org/nix/install) --daemon       # instalador oficial
```
O instalador criou os usuários de build `nixbld1..32` (uid 30000+), `/etc/profile.d/nix.sh`
(nix no PATH para **todos** os usuários) e `/etc/nix/nix.conf` com `build-users-group`.

### ARMADILHA: `LD_LIBRARY_PATH` global quebra o nix
O `setvars.sh` do oneAPI, chamado no `~/.bashrc`, exporta um `LD_LIBRARY_PATH` enorme
(inclusive `/usr/lib/x86_64-linux-gnu`). Isso **atropela o RPATH** do nix, que então
carrega a glibc/openssl do Ubuntu e falha:
`nix: /usr/lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.38' not found`.
Note que o token `LD_LIBRARY_PATH` **não aparece** no bashrc -- quem seta é o
`setvars.sh` internamente (grep pelo nome da variável não acha; grep por `intel`/`oneapi` acha).

Conserto aplicado **só na nossa conta** (`~/.bashrc`, backup em `~/.bashrc.bak-preoneapi`):
o export global foi desativado e o oneAPI virou **opt-in**:
```bash
use_oneapi() { ( . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; "$@" ); }
# uso:  use_oneapi python treino.py     (o shell externo segue limpo)
```
Verificado: com ambiente limpo, `nix --version` funciona e `echo ${LD_LIBRARY_PATH:-X}`
dá vazio (antes da correção vinha a lista inteira do oneAPI).

### Exposição que NÃO é nossa (reportar, não mexer)
- `/etc/skel/.bashrc:120` tem o mesmo `source .../setvars.sh` -> **toda conta nova herda**
  a quebra do nix. Correção exige root e é decisão de política do dono da máquina.
- `/home/fbonani/.bashrc:120` idem (outro usuário). Não é nosso para editar.
- Nada em `/etc/profile`, `/etc/bash.bashrc`, `/etc/environment`, `/etc/profile.d/` (verificado).

### Por que os jobs em Slurm não sofrem disso
Shell não-interativo **não lê** `~/.bashrc`. Por isso o bundle portable e o `srun`
funcionaram o tempo todo; só o uso interativo do nix precisava do conserto.

### Boa vizinhança com o store
`/nix/store` é **compartilhado** e cresce. Depois de builds pesados:
`nix store gc` (ou `nix-collect-garbage -d`). O store chegou a ~19 GB durante o
primeiro `nix develop` porque o nosso flake arrasta a toolchain **ROCm/hipBLAS**
(a fermi tem GPU; a halfbeast não) -- daí a ideia de um `devShells.cpu` enxuto.

### Wrapper para shells que ainda têm a poluição
`~/bin/nixc` = `exec env -u LD_LIBRARY_PATH /nix/var/nix/profiles/default/bin/nix "$@"`.
Útil em shells antigos/heredados; desnecessário em ambiente limpo.

### Compilar NATIVO agora funciona (o motivo de instalar o nix)

```bash
# na halfbeast, depois do nix novo:
cd ~/autoresearch-eval/autoresearch/src          # o fpm.toml mora em src/
~/bin/nixc develop .. --command fortran-fpm build      # 1o run baixa a closure
~/bin/nixc develop .. --command fortran-fpm test       # ~30 s, OMP_NUM_THREADS=2
```
Verificado (11/set 23h): build nativo OK e **suíte com 0 failures** em Skylake-X --
verificação cross-ISA dos nossos kernels (atenção BLAS fwd/bwd, tokenizer byte-level,
validação de arquitetura, save/load, amostragem) numa microarquitetura diferente da
fermi, onde foram escritos. Binário em `src/build/gfortran_<hash>/app/`.

Consequência para o fluxo: `bin/sync_eval.sh` (enviar binário) continua sendo o
caminho para *medir comparável* entre caixas (mesmo binário, mesma OpenBLAS, só a
CPU difere). Compilar nativo serve para (a) verificação cross-ISA, (b) velocidade
quando TODOS os braços de um experimento rodam nesta caixa -- misturar binário
nativo com binário enviado pode dar bpb diferente na última casa (-march diferente
muda ordem de soma), e isso quebraria a comparabilidade que a gente preserva.

### Verificação de boa vizinhança feita (11/set 23h)
- `fbonani`: em shell de login tem `LD_LIBRARY_PATH` **vazio**, nix funciona (2.35.2) ✓
- mtimes provam que não tocamos em nada alheio: `/etc/skel/.bashrc` e
  `/home/fbonani/.bashrc` seguem de **2022-05-06** ✓
- `/etc/profile.d/nix.sh` + `/etc/nix/nix.conf` existem -> nix disponível para todos ✓
- jobs rodando normalmente (`6010` do vizinho, `6016_0/1` nossos) ✓
