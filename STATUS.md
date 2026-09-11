# STATUS — sessão de 11/set/2026 (handoff para pós-compactação)

Estado vivo, números medidos, decisões abertas. Tudo abaixo foi verificado nesta
sessão; o que está marcado como estimativa é estimativa.

## O que está rodando AGORA

| onde | o que | termina |
|---|---|---|
| **halfbeast** (job Slurm 6010) | prosa v2: 2000 passos, linhas **1000..59384** (inéditas), `--attn blas`, 10 threads | ~22:00 |
| **fermi** (nohup, `logs/train_prose_v3.log`) | prosa v3: 2000 passos, linhas **3000..59384** (bloco seguinte), `--attn blas`, 8 threads | ~22:00 |
| **repl** (tmux `0:prose`) | inferência interativa no `best` da fase 5, template de chat | interativo |

Ambos partem de `/tmp/w_prose_final` (best da fase 5, 90/90 + template.txt) e
validam nas **mesmas 4950 linhas de livros held-out** — curvas comparáveis.
Rollback/limpeza: `pkill -f train_run` (fermi), `scancel 6010` (halfbeast).

## Fase 5 (prosa) — resultados

- **bpb em livros held-out: 6,69682 → 2,40226** (fase 4 → fase 5), 2,8×. Auto-teste
  do agregador: 2,72562 (meu) vs 2,72558 (treinador) ✓.
- Curva de val: 3,06640 (100) → 2,72558 (300) → 2,64492 (400) → 2,58930 (500) →
  2,50302 (700) → 2,41718 (900) → 2,40226 (1000), **ainda caindo** (subtreinado).
- Fase completa em **167 min** no halfbeast (BLAS) vs ~14,3 h no fermi (naive).
- `--attn blas` é **idêntico ao naive**: curvas de val batem em 5 decimais entre
  máquinas e kernels; NLL por passo idem.
- Gap val−trn pequeno (0,03) → **sem overfitting**; o problema é falta de passos
  (a fase viu 1,7% do corpus: 1000 linhas de 59.385).

## DOIS BUGS ENCONTRADOS E CORRIGIDOS (o mais importante da sessão)

1. **`f14a16a` — espaço de ids errado na inferência.** Os corpora são byte-level
   (`ASCII→byte`, `não-ASCII→256+b`, BOS 8188), mas `chat_text`/`repl` usavam o
   tokenizer **BPE** legado (`ranks.txt`): "A tarde caia" entrava como 6 ids
   (`65 261 276 356 1360 1137`) em vez de 13 bytes. O prompt **nunca chegava**;
   ids ≥256 saíam como tokens alheios (`428 → "home"`); ≥8188 imprimiam `<?>N`.
   ASCII sobrevivia por acidente (ranks 0..127 = bytes 0..127), e é por isso que
   o bug se escondeu. Conserto: `encode_bytes`/`decode_bytes`/`id_to_byte`
   (sem tabelas, inversos; ids não-texto descartados), testes `test_byte_space`.
   **Toda conclusão qualitativa anterior a esse commit é inválida** (incluindo os
   textos das baterias). **Todos os números sobrevivem** (bpb/NLL/md5 são em ids).
2. **`2cadc20` — máscara de ids válidos.** ~7800 dos 8192 ids nunca aparecem em
   corpus nenhum (só 4 têm byte-length zero) e eram amostráveis. `apply_byte_mask`
   + `sample_next(..., byte_space=.true.)` tornam-nos inalcançáveis. A primeira
   aplicação esqueceu **os dois call sites da verificação especulativa** — isso
   quebrava a equivalência spec↔simples; corrigido e revalidado (md5 idêntico
   para `--spec 0/8` nos dois apps).

## Performance (medida, não estimada)

- `attn_sgemm`/`attn_bwd_sgemm`: **12,98×** no kernel (7,9 → 102,8 GFLOP/s);
  FD test do backward: dQ 4,3e-04, dK 8,3e-04, dV 3,7e-04.
- Treino: **18,8×** numa corrida de 6 passos (1269 → 68 s), perdas idênticas.
- `eval_bpb --attn blas`: **10,1×** (75,7 → 7,5 s em 3 linhas), bpb idêntico.
- **Hiperthreading atrapalha** nos dois boxes: usar núcleos físicos (fermi 8,
  halfbeast 10). Launchers já usam `--attn blas` por padrão.
- Uma fase de 1000 passos: **~2 h** (era ~14 h).

## Decisão aberta: como unificar v2 e v3

Não é A/B — é treino em dados disjuntos, dois ramos do mesmo pai. Opções:

1. **Sequencial** (mais simples): continuar de um ramo usando o bloco do outro.
   Nada a unificar; só serializa o tempo.
2. **Merge de pesos (soup)**: média dos dois checkpoints (mesmo pai, mesmos
   hiperparâmetros → caso favorável). É o que o DeepSeek-V4.1 faz para agregar
   RL paralelo. Momentos Adam não são média: copiar de um ramo ou zerar.
3. **Dados intercalados** (correto para o futuro): um único arquivo com os dois
   blocos embaralhados e um único treino. Parallelismo só para wall-clock.

**Experimento para decidir** (barato, ~10 min): medir bpb em livros held-out nos
três — ramo A, ramo B e soup(A,B) — e passar as baterias no vencedor. Se o soup
ganhar dos dois pais, o paralelismo vira multiplicador de throughput legítimo.

## Instrumentos (onde achar as coisas)

- Corpus: `/tmp/prose/prose_v2.txt` (63335 linhas), `prose_v3.txt` (61335),
  `prose_all.txt` (original, 64335), `prose_val60.txt`/`prose_val300.txt`.
- Baterias: `bin/eval_infer.sh --weights DIR [--chat PATH] [--tables DIR]
  [--only core|residue|chat|security|multilingual|prose|all]`; a de prosa imprime
  `novelty4g`/`max_repeat_run` (n≥64, senão a métrica não diz nada).
- Eval no box quieto: `bin/sync_eval.sh`, `jobs/halfbeast_*.sbatch`
  (`_gate`, `_bpb`, `_attn_step`, `_attn_ab`, `_prose_continue`), logs em
  `bin/halfbeast_logs.sh`.
- Agregador de bpb: `scripts/bpb_from_eval.py` (reproduz o `val @N` do treinador).
- HEP: 34 hipóteses. Relevantes: `hyp_1854b7` TRAIN_THROUGHPUT 0,97;
  `hyp_191851` TOKEN_SPACE 0,95; `hyp_1fbfd5` PROSE-PT 0,66;
  `hyp_5ca396` MIXED-FINAL 0,60 (fase mista — o teste de esquecimento mostrou
  math→fortran→prosa apagando comportamento anterior); `hyp_ff4182` SAMPLE-MASK
  0,45 (kernel `causal_attn_doc` pronto e bit-exato; falta o `attn_bwd` mascarado
  — é o único pedaço de kernel ausente).

## Próximos passos, em ordem

1. Às ~22h: comparar as val de v2/v3 (devem cair; idealmente iguais entre si) e
   rodar o experimento de merge acima.
2. **Fase mista** (`hyp_5ca396`): intercalar math + fortran + prosa, mesmo
   orçamento de passos — recuperar o `core` sem perder prosa.
3. **`attn_bwd_doc`** para o A/B do `--docmask` (hoje quase irrelevante para
   prosa — 296/300 linhas sem BOS — e relevante para os corpora de instrução,
   onde 92% dos pares de atenção são cross-documento).
4. Retomar o **spec-decode com penalidades** (hoje bloqueado por design: com
   penalidades o argmax da linha não é a escolha do alvo; a solução é aplicar
   penalidade linha a linha na verificação — o mecanismo já existe em
   `fortran_spec_mod`).
