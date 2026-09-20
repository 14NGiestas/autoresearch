# Hypotheses of this lab

Derived from `hep/registry.jsonl` by `scripts/hep_doc.py`. Do not edit by
hand: run the script. A document and a registry that disagree leave no way
to know which one is wrong.

58 hypotheses. States: dormant 10, proposed 33, refuted 3, supported 6, under_test 6.

## hyp_ea9457 -- under_test, belief 0.5

DiLoCo-like (Adam interno + otimizador EXTERNO Nesterov sobre os parametros sincronizados a cada H passos) preserva a qualidade de um run unico de compute igual, enquanto fedavg puro com tau finito nao preserva; o termo de composicao e funcao do DESLOCAMENTO dos workers (lr x tau x disjuncao), nao do numero de shards.

* evidence: 14 (9 support, 5 refute)
* testable: 1) Braco DiLoCo no mesmo driver: inner Adam lr 6e-4, outer Nesterov lr ~0.7 com weight decay, H em {122, 488}, K=4, compute total fixo 7808 passos; comparar com fedavg puro (job 89) e com o run unico 2.120. 2) Varredura de lr do shard mantendo tau fixo: se o imposto for funcao do deslocamento, deve cair ~com lr^2. 3) Medir ||w_k - w_init|| por worker e correlacionar com o imposto par a par.
* mechanism: refine
* parents: hyp_ce049e

## hyp_fb3762 -- refuted, belief 0.1

QK-ROUTING: atencao sem value (2 hops sobre o stream) treina no nivel do baseline

* evidence: 3 (0 support, 3 refute)
* mechanism: de-novo
* sources: run.log

## hyp_5aab09 -- proposed, belief 0.62

TURING-PROSA: discriminacao cega humano-vs-modelo em continuacao literaria como funcao da escala; juiz 50%=indistinguivel; baseline P2-A (~85% detectavel esperado); curva com 25M; responde onde mora a inteligencia

* evidence: 13 (4 support, 2 refute)
* mechanism: de-novo
* parents: hyp_b6e33a
* sources: run.log

## hyp_67b9ad -- proposed, belief 0.64

SIZE-AXIS: o bpb nao melhora com o tamanho na nossa fatia (d96 4.542 / d216 4.496 / d360 4.724, tokens DESIGUAIS) porque o eixo esta CONFUNDIDO com o orcamento de tokens, e nao porque a lei de escala falha. Com tokens fixos e a receita vencedora, a serie controlada mostra a melhora na razao alpha das leis publicadas.

* evidence: 5 (3 support, 2 refute)
* testable: serie controlada de tamanho com tokens fixos e a receita tau=122 reset+outer: bpb cai monotonicamente com o tamanho, na razao alpha ajustada com erro <= 2%
* mechanism: inspired-by
* sources: arXiv:2304.01373 (Pythia); arXiv:2203.15556 (Chinchilla); arXiv:2402.00838 (OLMo); arXiv:2406.12907; HEP: inventario halfbeast (d96/d216/d360)

## hyp_688b63 -- supported, belief 0.72

TOK-LEX: intervencoes de tokenizacao movem H2 (taxa neologismo): BPE-dropout p=0.1 (barato, robustez) e MorphBPE (morfologia PT) reduzem H2 vs BPE-8K no testbed; 8K confirmado otimo; byte-level morto

* evidence: 15 (13 support, 1 refute)
* mechanism: inspired-by
* parents: hyp_5aab09
* sources: run.log

## hyp_167043 -- proposed, belief 0.92

SOUP-BASIN: a media de pesos (soup) so ganha do melhor pai se as branches partilham init (mesma bacia); inits diferentes -> bacinos independentes (drift ~raiz2) e media invalida

* evidence: 14 (13 support, 1 refute)
* mechanism: merge
* parents: hyp_5ca396
* sources: run.log

## hyp_fa2388 -- under_test, belief 0.82

FORTRAN_TRAIN_RUN: multi-batch training run in pure Fortran (cycling packed rows, val-bpb tracked per N steps) improves or holds depth-12 val bpb without divergence over 50+ steps

* evidence: 14 (11 support, 1 refute)
* mechanism: de-novo
* sources: src/app/train_run.f90; logs/train_run20.log; logs/train_overnight60.log; logs/train_resume40b.log; logs/train_lowr40.log; src/lib/fortran_sys.f90; logs/train_long100.log; bin/eval_infer.sh; bin/train_phase4.sh; logs/train_phase4.log, src/app/train_run.f90:--trn_probe; jobs/halfbeast_attn_step.sbatch (job 6006), src/lib/fortran_attn.f90, src/lib/fortran_train.f90

## hyp_1fbfd5 -- proposed, belief 0.45

PROSE-PT: a Portuguese prose phase (Project Gutenberg 645 PT books + MEC complete Machado de Assis, 71 PDFs -> 614 cleaned books, split BY BOOK: 558 train / 56 val, 134 MB) raises held-out-book bpb and the multilingual battery's surface competence WITHOUT regressing core in-dist math beyond noise. Rationale: we are data-starved (a phase = 2M tokens = 1.3% of this corpus) and our corpora are instruction/reasoning-shaped, so nothing teaches long-form coherence; byte-level tokenizer makes PT nearly free (measured 0.96 tok/byte PT vs 1.00 EN, no vocab surgery). Falsifiable: core battery must not fall; pt-gcd/pt-prime/es-gcd must move off zero; residue probes must stay plausible.

* evidence: 7 (2 support, 1 refute)
* testable: train_run on packed prose rows (2049-id contract) with val = 56 held-out BOOKS (never chunk-split, so no neighbour-paragraph leak); gate = bin/eval_infer.sh all five batteries + bpb on held-out books, compared against the phase-4 checkpoint
* mechanism: inspired-by
* sources: logs/train_phase5_prose.log, jobs: bin/chain_phase5_prose.sh; halfbeast job 5999, batteries_step_1000.txt; job 6003/6005, jobs/halfbeast_bpb.sbatch; job 6009, gate_phase4.txt/gate_phase5.txt; src/lib/tokenizer_encode.f90:encode_bytes, test_kernels.f90:test_byte_space; run.log

## hyp_ce049e -- proposed, belief 0.85

O imposto de composicao C(K) = bpb(media de K shards) - bpb(run unico, mesmos tokens totais) segue uma lei identificavel em K (saturante 1-1/K, logaritmica log K, ou de encolhimento 1-1/sqrt(K)); e a maior parte do imposto e explicada por ENCOLHIMENTO da norma da media, de modo que reescalar a media o recupera.

* evidence: 7 (4 support, 1 refute)
* testable: 1) C(K) para K=1,2,4,8 com orcamento total fixo; ajustar as tres formas e comparar residuos. 2) Medir s(K)=||media||/||shard|| e correlacionar com C(K). 3) Varrer a escala da media (0.8..1.2) sobre os checkpoints ja existentes: se C cair, a causa e encolhimento, nao destruicao de conhecimento. 4) Repetir com orcamento POR SHARD fixo (K maquinas em paralelo): baseline vira o run unico com K*T tokens. 5) Pior caso alem da media: max_i (bpb(media) - bpb(shard_i)) e altura da barreira ao longo do caminho linear entre pares.
* mechanism: refine

## hyp_fe4afe -- proposed, belief 0.7

MUON-SMALL: Muon (lr canonico 0.02) supera Adam em modelos 3M/8M toks no mesmo orcamento

* evidence: 6 (4 support, 1 refute)
* mechanism: de-novo
* sources: run.log

## hyp_87d151 -- dormant, belief 0.25

DATA-AUG: normalize + add claude/gemini/chatgpt raw shares (currently skipped: claude raw is error-JSON, gemini is HTML, chatgpt is dict-mapping) to the training corpus. Hypothesis: more diverse web-share text lowers val_bpb / improves generalization over opencode+pi only.

* evidence: 5 (1 support, 1 refute)
* testable: val_bpb lower than baseline after same budget
* mechanism: refine
* parents: hyp_bdc304
* sources: logs/2026-08-29/hoard_ultrachat.log; run_data_aug.log; git log 6b3ad1f; git log 7dfa54b; ai-chat-hoarder@a85704b

## hyp_bdc304 -- under_test, belief 0.45

BASELINE: run train.py as-is (DEPTH=5, TOTAL_BATCH=2**17, MAX_SEQ=2048, 12h budget) pretraining from scratch on hoard corpus (opencode+pi, ~65M tok, 88.7k docs). Establishes reference val_bpb.

* evidence: 4 (3 support, 1 refute)
* testable: val_bpb after 12h < 2.0 (model learns agentic distribution)
* mechanism: de-novo
* sources: run.log; run.log training curve (loss 9.0->1.0) + OOM traceback at train.py:108/137/298 + fix in train.py final-eval block

## hyp_25a8e7 -- dormant, belief 0.3

Adopt Qwen3.8-Flash-Next Muon refinements (orthogonalization accuracy, Muon/AdamW division of labour, split fused parameters, refit scaling law) in our from-scratch GPT to improve convergence and training stability at equal compute.

* evidence: 3 (1 support, 1 refute)
* mechanism: inspired-by
* parents: hyp_bdc304
* sources: train.py env-gated MATRIX_LR/WEIGHT_DECAY + bin/run_muon_refine.sh; run.log; git log 6b3ad1f

## hyp_2ac980 -- under_test, belief 0.95

Onde a fronteira 'recomputar vs. ler' o KV-cache realmente fica depende da razao compute:IO da maquina, nao do nivel de memoria: em CPU (com NVMe de 2.3 GB/s e RAM de 20 GB/s) ler o KV de 1 token custa 0.15 us (RAM seq), 8.5-10.4 us (RAM 1 IO/token), 1.3 us (disco seq) e 105-107 us (disco 1 IO/token, limitado por latencia); recomputar a janela custa 100 us/token (N=32), 140 (N=128) e 308 (N=512), porque a atencao e' O(N) por token. Logo em CPU ler vence SEMPRE contra RAM (10-2000x) e vence contra disco tambem, exceto janelas < ~50 tokens no regime 1 IO/token; se as leituras forem em LOTE, ler vence por 22-204x. Consequencia: a tecnica de fronteira de 'deletar a memoria local e recomputar 128 tokens' e' uma afirmacao sobre o razao FLOPs/IO (GPU tem ~100x os FLOPs da CPU com a MESMA latencia de NVMe), nao uma verdade universal.

* evidence: 3 (2 support, 1 refute)
* testable: medir em GPU o mesmo par (us/token recompute, us/token read por nivel) e ver se o N* sobe ~100x (de ~50 para ~5000 tokens)
* mechanism: de-novo

## hyp_9f47fd -- dormant, belief 0.3

Pre-tokenize the hoard to a flat .bin with an mmap dataloader (llama2.c style) to remove the CPU data-loading bottleneck (current mfu 0.2 percent, dt about 75s per step, 1736 tok per sec), raising effective throughput so more epochs fit the time budget.

* evidence: 3 (1 support, 1 refute)
* mechanism: refine
* parents: hyp_bdc304
* sources: train.py AUTORESEARCH_DATA_BIN + novel/bin_dataloader.py; run.log; git log 6b3ad1f

## hyp_cb3f6f -- dormant, belief 0.15

SMALL-MODEL GENERALIZATION: train a smaller GPT (DEPTH 5-6, n_embd 384, ~12-15M params) from scratch on the same ~400M-token hoard. The far higher tok/param ratio (~25-30 vs 0.4 for DEPTH=14) should force generalization (coherent, non-repetitive generation) instead of the memorization-collapse that broke the large model.

* evidence: 3 (1 support, 1 refute)
* mechanism: refine
* parents: hyp_bdc304
* sources: run_small.log; git log 6b3ad1f

## hyp_8f9016 -- supported, belief 0.75

O ganho de compor replicas satura em K~4 nesta escala: dobrar K (4->8) com o MESMO wall-clock e 2x o compute nao muda o bpb final (K=8: 2.32762 vs K=4: 2.33183, delta -0.004, dentro do ruido), e a fracao capturada do ganho ideal CAI (K=4: ~52% do caminho ate o run unico de mesmo compute; K=8: ~28%). Nao e' rede (o sync custa ~4 KB/token) nem otimizador: e' que K workers em fatias disjuntas passam a ver um lote efetivo que nao cresce com K.

* evidence: 2 (1 support, 1 refute)
* testable: medir a fracao capturada em K=2 e K=16 no mesmo wall-clock; se cair monotonicamente, o gargalo e' o lote efetivo (e entao K maior exige mais passos por worker, nao mais workers)
* mechanism: de-novo

## hyp_ba98cd -- proposed, belief 0.3

RECURRENT_DEPTH: a weight-tied looped block in our Fortran engine shows depth extrapolation (more inference loops solve deeper compositions) and overthinking (excess loops degrade), replicating Kohli et al. grokking dynamics at 17-97M scale

* evidence: 2 (1 support, 1 refute)
* mechanism: de-novo
* sources: src/lib/fortran_recurrent.f90; logs/recur_sweep_8rows.log:src/app/recur_sweep.f90

## hyp_00da77 -- refuted, belief 0.15

TRASH2GOLD_WRAP: WRAP-style rephrase hoard (400M diverse trash) into textbook/QA gold via LLM rephrasing (Mistral-7B-Q4 / Qwen style) then pretrain depth-12 97.5M on rephrased 400M. Phi-1 showed 6B textbook+1B synthetic beats 10x larger; WRAP showed C4 rephrased beats HQ. Tests if style transfer on same tokens breaks memorization (France France loops) and yields chat.

* evidence: 1 (0 support, 1 refute)
* testable: val_bpb on rephrased holdout < 0.5 and chat 'write python script' yields non-repetitive code vs base 0.375 memorizing
* mechanism: inspired-by
* parents: hyp_4b57c1, hyp_cb3f6f
* sources: run_wrap_gold.log + checkpoint_depth12_step265_na.pt

## hyp_6031ab -- proposed, belief 0.15

O passo de treino e' dominado por kernels ELEMENTWISE escalares (rmsnorm, rope, relu^2, lacos da atencao/loss), nao pelos GEMMs: com 14 GFLOP/s = 2-4% do pico e throughput plano em B, vetorizar/BLAS-izar os elementwise e' o caminho para o speedup (nao lote, nao composicao).

* evidence: 1 (0 support, 1 refute)
* testable: medir tempo por FASE/kernel no train_run (fwd vs bwd vs update; e rmsnorm/rope/relu2/atencao isolados) e comparar com uma versao vetorizada; se os elementwise somarem >50% do tempo, o ganho esta' ali
* mechanism: inspired-by

## hyp_a45a4b -- proposed, belief 0.08

WASI-SUBSPACE: treino from-scratch vive em subespaco fixo pequeno (corte 62x memoria => ~1.6% das dims conteriam a trajetoria)

* evidence: 1 (0 support, 1 refute)
* mechanism: inspired-by
* parents: hyp_5ca396
* sources: run.log

## hyp_c3aed6 -- proposed, belief 0.2

O maior ganho disponivel para o objetivo de speedup (16 anos -> 6 meses) nao esta em composicao (medida: 1.35x de eficiencia em K=4, caindo com K) e sim em EFICIENCIA DE PASSO: o treino roda com B=1 fixo em tempo de compilacao (integer, parameter :: B = 1 em train_run.f90), o que faz cada GEMM ser minusculo (M=1024, N=K=d=96 -> 0.019 GFLOP) e rende apenas ~30 GFLOP/s em ~8 cores (~5-10% do que os cores fazem com formas boas). Medido no log do worker (k1_f0, rodada 122 passos): 124928 tokens em 68.5 s = 1822 tokens/s de 8 threads. Subir B ate a eficiencia saturar multiplica ATUALIZACOES por unidade de wall-clock sem gastar comunicacao nem mudar a matematica; o limite honesto e' o tamanho critico de lote (mais tokens por update = menos updates por token), entao o alvo nao e' 'B maximo' e sim o B que maximiza (updates/s) x (qualidade/update) a tokens fixos.

* evidence: 1 (0 support, 1 refute)
* testable: B como parametro de build entrando na identidade da arch; depois varrer B em {1,4,16,64} a tokens fixos medindo tokens/s (energy_trace.csv) e bpb. Se tokens/s subir >=3x e o bpb a tokens fixos perder <=0.05, o ganho de wall-clock e' maior que todo o ganho de composicao somado.
* mechanism: de-novo

## hyp_c649cc -- proposed, belief 0.35

DEPTH14_GC: depth 14 (896-d=~97M but actually 122M at n_embd=896) with GRAD_CHECKPOINT=1 + expandable_segments to fit 16GB. Depth12 proved 97.5M generalizes without collapse (step264 val_bpb 0.375). Depth14 OOMed without GC at step1. With GC we trade ~30% speed for ~40% VRAM saving, should fit. Tests whether pushing to largest vanilla that fits improves val_bpb <0.35 and generation coherence.

* evidence: 1 (0 support, 1 refute)
* testable: Fit check: does step1 succeed without OOM. Then loss/val_bpb at step100 vs depth12@100. If val_bpb <0.35 and generation more coherent than depth12 greps, supports scale hypothesis toward 1B.
* mechanism: refine
* parents: hyp_f3adc8, hyp_4b57c1
* sources: logs/2026-08-29/run_large_gc.log

## hyp_ff4a76 -- proposed, belief 0.25

Chinese-literacy regime: train a from-scratch GPT with a Chinese-aware mixed tokenizer on a combined English-agentic + Chinese-text corpus, steering the model to reason in dense Chinese while emitting English agentic tool syntax. Mechanism: Chinese packs more information per token, raising effective tok/param and directly attacking the memorization-collapse root cause; English tool strings are frozen so capability (get-shit-done) is preserved. Prereqs: (a) a Chinese-capable tokenizer (current 8192 English-only vocab fragments Chinese into MORE tokens, negating compression) and (b) Chinese-thought/English-action parallel data (hoard Chinese agentic text or LLM-generated).

* evidence: 1 (0 support, 1 refute)
* mechanism: refine
* parents: hyp_cb3f6f
* sources: https://markhuang.ai/blog/chinese-token-myth

## hyp_34ea7c -- supported, belief 0.99

FORTRAN_TRAIN_FULL: port entire training loop to Fortran (mfi/fpm + GPU kernels via ZLUDA/HIP) — forward (rmsnorm/rope/attn/mlp already in novel/fortran_math.f90) + backward + MuonAdamW. Use mfi's GPU support or write zluda HIP kernels. Tests if Fortran training matches torch numerics and keeps 24/7 VRAM hot with native kernels.

* evidence: 22 (22 support, 0 refute)
* testable: fortran_gpt_forward error <1e-6 and training loss trajectory matches torch within 1% for 50 steps on data_wrap_gold
* mechanism: inspired-by
* parents: hyp_00da77
* sources: novel/fortran_math.f90 + novel/fortran_math_kernels.so; src/test_kernels; src/test/test_kernels.f90:/tmp/parity.py; https://github.com/cambridge-ICCS/FTorch; scripts/export_weights.py:src/app/eval_bpb.f90:scripts/eval_driver.py; src/lib/tokenizer_encode.f90:scripts/tokdiff_driver.py; src/app/chat_text.f90; logs/tokdiff_1000.log; logs/fortran_bpb_48rows_v2.log; logs/fortran_bpb_normfix.log; src/lib/fortran_backward.f90; src/lib/fortran_attn.f90; src/lib/fortran_adamw.f90; src/lib/fortran_train.f90; src/app/train_1step.f90:src/lib/fortran_data.f90; src/app/train_loop.f90; logs/overfit_10step.log; logs/overfit_10step.log:logs/fortran_bpb_normfix.log; src/lib/fortran_blas.f90; src/lib/fortran_gpt.f90

## hyp_842fba -- proposed, belief 0.9

SPEC_DECODE: greedy draft-and-verify speculative decoding for CPU inference. Draft k tokens with a depth-3 drafter (truncate existing checkpoint: wte + layers 0-2 + lm_head), verify in ONE full-model forward. Rationale: our decode is memory-bandwidth-bound at B=1 (weights streamed per token), so verifying k tokens costs ~1 token's time. DeepSeek-V4.1 does this at scale (DSpark: 3-block SWA drafter, trained post-hoc with frozen backbone). Prediction: acceptance >=50% on in-distribution prompts at temp 0, end-to-end decode tok/s >=1.5x vs current gpt_step loop.

* evidence: 6 (6 support, 0 refute)
* testable: chat_text --stats tok_s with/without draft-verify on a fixed prompt set; acceptance rate printed; no retraining needed for the pilot (drafter = truncated phase-4 checkpoint)
* mechanism: inspired-by
* sources: src/lib/fortran_attn.f90:attn_chunk, src/lib/fortran_kv.f90:gpt_step_multi, src/lib/fortran_spec.f90, src/test/test_kernels.f90:test_attn_chunk,test_kv_chunk_equiv,test_accept_prefix; src/app/spec_bench.f90, src/lib/fortran_spec.f90, src/test/test_kernels.f90:test_accept_prefix,test_lookup_draft; src/app/chat_text.f90:--spec, src/lib/fortran_spec.f90:lookup_draft; src/app/repl.f90:--spec, src/app/chat_text.f90:spec stats; src/app/chat_text.f90, src/app/repl.f90, src/lib/fortran_spec.f90; jobs/halfbeast_eval20.sbatch, logs eval_t20_5998

## hyp_d2c4dc -- proposed, belief 0.93

SOUP-PROXIMITY: ganho da sopa exige drift pequeno (<<1), nao so mesmo init; curva ganho-vs-drift tem limiar entre 1e-4 (ganha) e 0.5 (perde +0.23)

* evidence: 5 (5 support, 0 refute)
* mechanism: refine
* parents: hyp_167043
* sources: run.log

## hyp_11df96 -- dormant, belief 0.58

Adding Qwen3.8-Flash-Next Gated Residual (4-branch data-dependent gated residual stream) to our from-scratch GPT improves training stability and final validation bpb at equal parameter/compute budget on the 65M-token hoard.

* evidence: 4 (2 support, 0 refute)
* mechanism: inspired-by
* parents: hyp_bdc304
* sources: novel/gr_model.py _cpu_self_test (25,600 real hoard tokens); train.py AUTORESEARCH_MODEL=gr + novel/gr_model.py; run.log; git log 6b3ad1f

## hyp_58f2c4 -- dormant, belief 0.5

Add Qwen3.8-Flash-Next N-gram Embedding (deterministic local-context lookup memory) to our from-scratch GPT to scale model capacity at negligible extra compute.

* evidence: 3 (2 support, 0 refute)
* mechanism: inspired-by
* parents: hyp_bdc304
* sources: novel/ngram_model.py + train.py AUTORESEARCH_MODEL=ngram; src/docs/deepseekv4.1.md:2.4.2; git log 6b3ad1f

## hyp_5ca396 -- proposed, belief 0.75

MIXED-FINAL: running the curriculum as a sequence of PURE phases (code -> tools -> math -> fortran -> prose) destroys the previous phase's behavior, and the fix is mixing, not ordering. Measured with the same battery on the same weights-scale: after the math phase, in-dist 'Find the GCD of 54 and 24' produced math-textured output ('5 545.'); after the fortran phase the SAME prompt produced nothing at all (whitespace) and 'is 97 prime' produced a row of dots ('...........'). The paraphrase case survived ('5?'), so it is not total erasure -- it is the in-distribution behavior that gets overwritten. Cognivolve (our curriculum's source) explicitly closes with a MIXED stage over all previous phases for this reason, and Beyond Random Sampling finds curriculum is strongest as a warmup followed by random/mixed sampling. Prediction: a mixed final phase (math + fortran + prose interleaved, same total steps) keeps prose/multilingual gains while restoring in-dist math ABOVE the pure-fortran baseline; a pure phase cannot, at any step count.

* evidence: 3 (2 support, 0 refute)
* testable: train the same 1000-step budget as a mixture (e.g. 1/3 rows from each phase's corpus, shuffled) from phase-4 step_1000, then run the six batteries: core must beat the pure-phase baseline and prose/multilingual must not regress
* mechanism: inspired-by
* sources: run.log

## hyp_e32525 -- proposed, belief 0.35

WISCA-CONT: continuar treino de checkpoint reescalonado (QK^T e (attn.V).Wo preservados, ponto mais plano) ganha do continuation ingenua; controle: mesmos moments frescos nos dois bracos; GQA-kv2 e o caso favoravel

* evidence: 3 (1 support, 0 refute)
* mechanism: inspired-by
* parents: hyp_5ca396
* sources: run.log

## hyp_1854b7 -- proposed, belief 0.97

TRAIN_THROUGHPUT: the hand-written attention kernels, not the hardware, were the training bottleneck. Measured on the idle i9 at 10 threads (same weights/rows/flags, only --attn differs, 6 steps + val + save): naive 1269.2 s vs blas 67.6 s = 18.8x for the whole run, with per-step NLLs IDENTICAL to 5 decimals (2.10027/1.87027/1.84340/1.97125/1.89796/1.83906 in both) and val/trn bpb identical (2.81231/2.65640). Subtracting the identical fixed costs (weight load + checkpoint save) and the val/probe rows (which also paid the naive kernel: 8 rows x ~45 s), the per-step estimate is ~147 s naive vs ~3-7 s blas -- i.e. a 1000-step phase that costs ~9 h drops to ~1.5-2 h, on the machine that LOOKED slower. Corollary: phase A/Bs (docmask on/off, mixed phase) stop being overnight commitments.

* evidence: 2 (2 support, 0 refute)
* testable: run a full 1000-step phase with --attn blas on the idle box and check wall time against the ~9 h naive baseline; the 6-step measurement already fixes the ratio (18.8x total, identical losses), only the extrapolation to 1000 steps is untested
* mechanism: de-novo
* sources: job 6007, ~/autoresearch-eval/phase5_blas_run.log; /tmp/fermi_ab.sh output, bin/run_curriculum.sh, bin/train_phase5_prose.sh

## hyp_6f5423 -- supported, belief 0.95

BACKEND-ST: o backend safetensors no trainer escreve/le checkpoints com fidelidade byte a byte (mesmos pesos, mesmos NLLs) e carrega a arch no __metadata__, substituindo os .npy sem perda.

* evidence: 2 (2 support, 0 refute)
* testable: 1) eval_bpb com --ckpt-format npy vs st no mesmo checkpoint: NLLs byte-identicos. 2) Round-trip escrevendo os dois formatos das mesmas matrizes e comparando bytes (nao valores). 3) Leitor Python independente (reference_writer) lendo o arquivo Fortran com hashes de payload iguais aos .npy. 4) Checkpoint REAL de experimento em both, nao sintetico.
* mechanism: de-novo

## hyp_ad78cc -- proposed, belief 0.95

O gargalo do passo e' a ATENCAO, e a causa raiz e' arquitetural: head_dim=16 faz as scores [T,T] dominarem com intensidade aritmetica pessima (4.5 GFLOP/s contra 40.3 GFLOP/s com head_dim=128 no mesmo T). Os ganhos disponiveis sao (a) atencao causal BLOCADA/flash (nao materializar a matriz T x T; o caminho BLAS atual faz o quadrado inteiro = 2x desperdicio) e (b) head_dim maior (menos cabecas, produtos internos mais largos) -- uma mudanca de ARQUITETURA justificada por medicao, nao por intencao.

* evidence: 2 (2 support, 0 refute)
* testable: medir (i) atencao blocada causal vs sgemm quadrado no mesmo T, e (ii) head_dim 16 vs 32 vs 64 a params fixos: se o passo cair >=2x sem perder bpb, o gargalo esta' confirmado e o caminho e' arquitetural
* mechanism: de-novo

## hyp_c30aeb -- dormant, belief 0.62

CLAUDE-STYLE: continue baseline weights and fine-tune on the Claude-normalized corpus (claude.ai exports, 422 sessions) to capture Claude's writing style; measured by val_bpb on a held-out Claude set (shard_06542 in data_claude).

* evidence: 2 (1 support, 0 refute)
* testable: val_bpb on Claude held-out set lower than baseline's claude-subset proxy; generated samples read like Claude.
* mechanism: refine
* parents: hyp_bdc304
* sources: run_claude.log,train.py; git log 6b3ad1f

## hyp_f3adc8 -- dormant, belief 0.55

LARGE_VANILLA: Train vanilla GPT depth=14 (n_embd=896, ~96M params) on COMBINED hoard (mixed 400M + ultrachat 515K) for 4h. The prior depth-14 baseline collapsed on the mixed hoard (garbage despite val_bpb 0.203). The data-aug run (16.9M params) also collapsed even on clean English data. The bottleneck may be BOTH: (a) model too small to learn coherent token distributions, and (b) data quality. A larger model (96M vs 17M) on combined data may finally break the memorization barrier.

* evidence: 2 (1 support, 0 refute)
* testable: Watch generation samples at step 50, 100, 200. If text becomes coherent (no ioio, tooto, ???, Serft loops), the larger model + combined data breaks the collapse. If still garbage, the training setup itself is the blocker (need more tokens, different architecture, or pretrained init).
* mechanism: refine
* parents: hyp_cb3f6f, hyp_87d151
* sources: logs/2026-08-29/run_large_bf16.log; git log 6b3ad1f

## hyp_035ebc -- supported, belief 0.85

ENERGY-CARD: o custo energetico por token e medivel neste lab (RAPL/hwmon por job) e pode viajar DENTRO do checkpoint (__metadata__), de modo que treino direto e composicao de K shards sejam comparaveis em dJ/dbpb e nao so em parede de relogio.

* evidence: 1 (1 support, 0 refute)
* testable: 1) Calibracao em maquina vazia: J/Mtok do d96 a 8 threads (fermi) e do mesmo run em outra arch. 2) card_annotate: juncao ledger->metadata com verificacao sha256 dos payloads intactos. 3) Aplicar aos runs do grid de composicao e ao fedavg/DiLoCo e reportar dJ/dbpb por braco. 4) Custo do imposto de composicao em kWh por unidade de perda, comparado ao ganho de wall-clock.
* mechanism: de-novo

## hyp_10996a -- dormant, belief 0.55

CAVEMAN REGISTER: fine-tune a generalizing base (small model hyp_cb3f6f) with FREEZE_BACKBONE=2 / low LR on caveman-register agentic-chat data, so the model emits caveman-style text (e.g. 'ME BUILD FIRE. YOU WATCH.') while retaining task capability (code/reasoning lives in the frozen backbone). Same style-finetune machinery as the Claude-style run (hyp_c30aeb), different target register.

* evidence: 1 (0 support, 0 refute)
* mechanism: refine
* parents: hyp_c30aeb hyp_cb3f6f
* sources: git log 6b3ad1f

## hyp_2ae064 -- proposed, belief 0.35

LARGE_BF16: Train vanilla GPT depth=14 at bfloat16 with DEVICE_BATCH=4 (vs baseline 2). bfloat16 halves weight/gradient memory, allowing a larger batch. Larger batch = more tokens per step = better gradient signal per step = potentially better generalization. Same 96M params, same data, but more tokens/step. If this still collapses, the training dynamics (optimizer, LR, regularization) are the bottleneck, not precision or batch size.

* evidence: 1 (0 support, 0 refute)
* testable: Watch generation at step 50, 100, 182. If coherent text at step 182+ while val_bpb stays reasonable, bfloat16+larger batch helps. If still garbage at step 182, the training setup itself (LR schedule, no regularization, data quality) is the fundamental blocker.
* mechanism: refine
* parents: hyp_f3adc8
* sources: logs/2026-08-29/run_large_bf16.log

## hyp_456a59 -- proposed, belief 0.55

GRAD-RANK: rank instantanea do gradiente << rank integrada do drift (subespaco RODA); e rank efetiva varia com dieta (f0 vs f100); medido via dumps de gradiente no testbed d96 + SVD offline; distingue fixo-vs-rotativo e mecaniza dieta-vence-escala

* evidence: 1 (0 support, 0 refute)
* mechanism: de-novo
* parents: hyp_a45a4b
* sources: run.log

## hyp_4b57c1 -- proposed, belief 0.65

DEPTH_12_FINAL: Train vanilla GPT depth=12 (n_embd=768, 97.5M params) on mixed hoard for full 4h budget. Loss at step 264 is 1.35 (val_bpb 0.375). If loss continues dropping and generation becomes coherent (no ioioio/amea loops), depth 12 proves sufficient scale. If it collapses like depth 5-6, the collapse is purely data-quality driven and we need either: (a) larger model, (b) pretrained init, or (c) different architecture (GR/Gated Residual). Test: generation at step 500 if trained to completion.

* evidence: 1 (1 support, 0 refute)
* testable: val_bpb at end of run. If <0.30 with coherent generation, hypothesis confirmed. If still fragmented/garlike depth 5-6, hypothesis refuted and the training setup itself is the blocker.
* mechanism: refine
* parents: hyp_2ae064
* sources: logs/2026-08-29/run_large_bf16.log

## hyp_5b7349 -- refuted, belief 0.3

CS_ROLEPLAY_15M: train tiny depth=4 (15M, SEQ 256) on data_cs_chat (12k synthetic event->chat from nvd_bot_chat templates, PT-BR 10 words) for crappy-hardware bot. Tests if small model can roleplay as player on CPU (50MB Q4, 20 tok/s) ingesting game events and outputting chat.

* evidence: 1 (0 support, 0 refute)
* testable: chat 'Killed BOT_Alex with AK 2v1' -> 'ez' style short PT-BR; runs on CPU with chat.py and fits docker-nvd
* mechanism: inspired-by
* parents: hyp_00da77
* sources: nix develop + train.py import failure

## hyp_708ad5 -- proposed, belief 0.9

NPY-ROWS: corpus rows como npy 2D unico (N,TT+1) int32 + json de regua (split hash, livros held-out): zero parse para treino/eval/agregacao, mmap, legivel em numpy e stdlib load_npy; texto passa a registro; indices-ancora sobrevivem (formato-agnosticos)

* evidence: 1 (1 support, 0 refute)
* mechanism: inspired-by
* parents: hyp_5ca396
* sources: run.log

## hyp_e4ab9d -- proposed, belief 0.3

CURRICULUM-DATA: ordenar corpus por dificuldade (comprimento/taxa-palavra-rara) easy->hard reduz steps-para-mesma-loss vs embaralhado; prior fraco pelo contra-dado de ordem (A vs B2: -0.005 com ordens independentes); custo ~zero, teste pequeno pos-M25

* evidence: 1 (0 support, 0 refute)
* mechanism: inspired-by
* parents: hyp_5ca396
* sources: run.log

## hyp_ef823b -- proposed, belief 0.85

ANCHOR-60: 60 linhas-ancora (gulosa por rank Spearman) predizem full-val 3985; k=60 e o joelho (k=100: +0.01pp); first-60/random empatam rank com 5 modelos (2 outliers dominam); resolucao de pares proximos (~0.2%) fica no ruido

* evidence: 1 (1 support, 0 refute)
* mechanism: inspired-by
* parents: hyp_5ca396
* sources: run.log

## hyp_f48678 -- supported, belief 0.9

EPOCH-COVERAGE: o deficit de K shards fundidos contra 1 maquina nao e falta de epocas, e sim de TOKENS DISTINTOS por worker. Previsao: repetir a fatia (mais epocas) melhora shard e run unico de forma parecida, entao o imposto de composicao nao fecha -- e pode CRESCER, porque cada shard fica mais seguro da propria fatia e a media vira compromisso pior.

* evidence: 1 (1 support, 0 refute)
* testable: 1) Grid de epocas: K=4 com 1/2/4 epocas da propria fatia (1952/3904/7808 passos por worker) + merge, contra runs unicos nos dados completos com o MESMO compute total (7812/15624/31248 passos) e o mesmo wall-clock. 2) Medir o imposto de composicao (merge - media dos shards) em cada nivel de epoca: previsao e que ele cresca com epocas no regime federado. 3) Comparar com o regime data-parallel (job 97), onde a previsao e o oposto (sync ajuda e o imposto cai).
* mechanism: refine
* parents: hyp_ce049e

## hyp_ff4182 -- proposed, belief 0.45

SAMPLE-MASK: mask attention across BOS document boundaries during training (block-diagonal causal mask), DeepSeek-V4.1-style 'sample-level attention masking'. Measured: only 8.1% of causal pairs in phase-3 math rows are within-document (12.5 BOS-marked docs/row) — 92% of attention capacity trains cross-document relations that never occur at single-doc inference. Prediction: mask-aware bpb on the pinned val shard improves vs full-causal at fixed steps, esp. on short-doc corpora.

* evidence: 1 (0 support, 0 refute)
* testable: eval_bpb (pinned val rows, masked identically in train+eval) vs current full-causal baseline at matched steps; needs per-row boundary metadata in the rows-file format (not exported today)
* mechanism: inspired-by
* sources: src/lib/fortran_attn.f90:causal_attn_doc, src/test/test_kernels.f90:test_causal_attn_doc

## hyp_191851 -- proposed, belief 0.95

TOKEN_SPACE: inference must use the SAME id space as the training corpora, and nothing enforced that. The corpora are byte-level (ASCII -> byte, non-ASCII -> 256+b, BOS 8188, per scripts/tokenize_corpus.py), but chat_text/repl were encoding prompts and decoding outputs through the legacy BPE tokenizer (ranks.txt): 'A tarde caia' went in as 6 BPE ids (261 276 356 1360 936) instead of 13 bytes, so the model never received a coherent prompt, and generated ids >= 256 were displayed as unrelated BPE tokens (428 -> 'home') while ids >= 8188 printed as '<?>8188'. ASCII survived by accident because ranks 0..127 are the single bytes in order, which is why output looked half-right and the bug hid for weeks. Fix: encode_bytes/decode_bytes (table-free, own inverse, unit-tested; non-text ids dropped instead of rendered). Consequences: every qualitative generation result before this fix is void (the eval batteries' text sections included); all METRICS stand, because bpb/loss/md5 operate on ids, identical on both sides -- which is exactly why the bpb self-check (2.37221 vs trainer 2.37217) passed while the text was garbage. Open question this raises: what ELSE differs between the two spaces (sampling mask for undefined ids, stop strings, the token_bytes weighting of ids the model can emit but the corpus never used)?

* evidence: 0 (0 support, 0 refute)
* testable: byte-level round-trip unit test (added to the suite) plus: encoded ids must equal scripts/tokenize_corpus.py output; and generated text must render accents correctly (it did not before)
* mechanism: de-novo

## hyp_269b5f -- dormant, belief 0.55

GRAD-RANK: rank instantanea do gradiente << rank integrada do drift (subespaco RODA); e rank efetiva varia com dieta (f0英语f100); medido via dumps de gradiente no testbed d96 + SVD offline; distingue fixo-vs-rotativo e mecaniza dieta-vence-escala

* evidence: 0 (0 support, 0 refute)
* mechanism: de-novo
* parents: hyp_a45a4b

## hyp_29ecb5 -- proposed, belief 0.45

LFortran compiles our Fortran GPT engine unmodified and reproduces GFortran numerics

* evidence: 0 (0 support, 0 refute)
* mechanism: de-novo

## hyp_54ad37 -- proposed, belief 0.9

INFER_COST: our CPU inference has a hard operating point that most of today's numbers violated. (1) Decode is memory-bandwidth bound: 48-token decode peaks at 10 threads (10 physical cores) at 1047 ms and hyperthreading HURTS it (16t=1147, 20t=1140 ms); prefill is GEMM/compute bound and also peaks at 10 physical cores (3545 ms, HT adds nothing, 20t=3233 ms). (2) Chunked prefill (--pchunk 64, attn_chunk+gpt_step_multi) cuts an 880-token prefill from 41875 to 3545 ms = 11.8x at 10 threads. (3) Therefore any tok/s or prefill_ms quoted at another thread count, or measured next to the 16-thread trainer, is not comparable -- fermi figures taken under load varied +-20%. Corollary already used: always set OMP_NUM_THREADS/OPENBLAS_NUM_THREADS explicitly (unset cost 42x: 0.87 vs 37 tok/s).

* evidence: 0 (0 support, 0 refute)
* testable: sweep OMP_NUM_THREADS in {2,4,8,10,16,20} on an idle box for (a) --pchunk 64 prefill of a fixed 880-token prompt and (b) n=48 greedy decode; the optimum must sit at the physical-core count and HT must not help decode
* mechanism: de-novo

## hyp_83bdd2 -- proposed, belief 0.95

EVAL_PORTABLE: an eval run is trustworthy across machines if the binary and its shared libraries travel together. fermi's chat_text plus the exact .so closure it needs (same OpenBLAS/glibc) plus an ld.so wrapper runs on halfbeast's i9-7900X with NO sudo and no system change (halfbeast's system nix is 2.6, too old for our pinned nixpkgs which needs >=2.18, and upgrade-nix cannot fix a distro install). Evidence: four prompt/seed fingerprints are md5-IDENTICAL across Ryzen 7 8745HS and i9-7900X while OpenBLAS dispatches different GEMM kernels per CPU, and md5 stays identical across --pchunk 1/64/256, --spec 0/8 and 2..20 threads. So halfbeast is a fidelity box, not a speed box: it is ~18% SLOWER than the 2023 Ryzen at 1-thread prefill (46.2s vs 38.9s).

* evidence: 0 (0 support, 0 refute)
* testable: run the same prompt/seed on both boxes with the shipped bundle and compare md5; any mismatch means ISA/BLAS, not noise
* mechanism: de-novo

## hyp_8ad675 -- under_test, belief 0.48

MOE_OFFLOAD_8E: 8-expert MoE 200M total (25M/expert, top-2 active 50M) with RAM offload (inactive 150M in host) on data_wrap_gold textbook. Tests if sideloading inactive experts to RAM (DeepSpeed-MoE style) lets 16GB APU train larger MoE as you described — same loop as dense but with expert swap. Expect same tok/s with swap overhead.

* evidence: 0 (0 support, 0 refute)
* testable: VRAM peak <14GB with 200M total vs dense 97M 12GB; tok/s ~1500 vs 2360; val_bpb <0.5 and expert utilization balanced
* mechanism: inspired-by
* parents: hyp_00da77, hyp_c649cc

## hyp_a4bc22 -- proposed, belief 0.45

CLAUDE_FT_DEPTH12: resume depth-12 97.5M (0.375) checkpoint and fine-tune on data_claude (6.4M, Claude style) with LR_SCALE=0.3 for 4h. Tests original goal: can a memorizing base still be steered to Claude writing style via low-LR fine-tune? If chat after FT produces Claude-like style without 'France France' loops, fine-tune rescues memorization.

* evidence: 0 (0 support, 0 refute)
* testable: chat 'write a python script' with FT checkpoint yields diverse code vs base regurgitation; val_bpb on claude holdout < 2.0
* mechanism: refine
* parents: hyp_4b57c1, hyp_c30aeb

## hyp_a8c643 -- under_test, belief 0.5

10k rows (vs 48) with true holdout 500 reduces trn/val gap and stops memorized CoT prefix (The user wants me to) — gap measured by new train_run logger trn/val bpb

* evidence: 0 (0 support, 0 refute)
* mechanism: de-novo

## hyp_b6e33a -- proposed, belief 0.65

PAIR-25M: par mesmo-init + janelas deslocadas (receita v2/v3) mapeia o limiar da SOUP-PROXIMITY; sopa iff drift<0.1; wall-time igual ao run unico (1 branch/box em paralelo)

* evidence: 0 (0 support, 0 refute)
* mechanism: refine
* parents: hyp_d2c4dc

## hyp_e6401b -- proposed, belief 0.45

JEV-INTERFACE: num modelo pequeno, uma cabeca de classificacao supervisionada (cross-encoder, objetivo de rotulo) extrai decisao calibrada muito acima do acaso onde a verossimilhanca condicional do LM gerador (o baseline 'structured output, um token por classe, prefill-only') fica no acaso; ou seja, o gargalo da decisao e o OBJETIVO/interface, nao a informacao no modelo.

* evidence: 0 (0 support, 0 refute)
* testable: 1) Probe zero-shot no ASSIN2-RTE (entailment binario, 2448 pares de teste, acaso 50%): ganho de NLL condicional ordena entailment? (job 6082). 2) Mesmo dado, mesmo encoder (d96/d216), cabeca de 3 classes treinada com CE: acuracia e ECE no teste. 3) Comparar com o harness generativo do ARC (argmin-NLL: acc 0.22, vies de posicao A=37%, ECE 0.05-0.08). 4) Se a cabeca supervisionada vencer com folga, medir a curva de escala (3M/10M/25M) para ver onde a decisao emerge.
* mechanism: inspired-by

## hyp_fa5c76 -- proposed, belief 0.35

Replace standard softmax attention in our from-scratch GPT with the Qwen3.8-Flash-Next GDN plus QSA hybrid (Gated DeltaNet recurrence with micro-block sparse attention) to enable long context at linear compute cost.

* evidence: 0 (0 support, 0 refute)
* mechanism: inspired-by
* parents: hyp_bdc304

