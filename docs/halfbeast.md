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
