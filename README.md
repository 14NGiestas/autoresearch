# autoresearch (AMD / torch-free fork)

> *One day, frontier AI research used to be done by meat computers… This repo is the story of how it all began. —@karpathy, March 2026.*

This is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
("give an AI agent a small LLM setup, let it experiment overnight").
Upstream targets a single NVIDIA GPU with PyTorch (`uv run train.py`,
5-minute budget, `val_bpb` metric).

**This fork diverged:** AMD Ryzen 7 8745HS + Radeon 780M (16 GiB carve-out),
`nix develop` instead of `uv`, and — as of Sep 2026 — **PyTorch is fully
ditched**. The model lives in **pure Fortran** (`src/`, fpm package):
forward pass, BPE tokenizer, and real-weight inference with zero Python
at runtime. See `program.md` for the current agent instructions and
`hep/registry.jsonl` for the experiment audit trail (21 hypotheses).

## What was proven here

- **Depth-12 from-scratch GPT (97.5M, n_embd=768) hit `val_bpb: 0.375`**
  (`checkpoints/checkpoint_depth12_step264_0.3750.pt`) — memorization→
  generalization threshold between 17M and 97M params on agent-chat corpora.
- **Fortran forward == `train.py` wiring to 2e-7** (numpy parity harness),
  10/10 kernel tests, GQA + RoPE + ReLU² + RMSNorm all exact.
- **BPE tokenizer byte-exact vs tiktoken** (200 docs / 778,724 tokens):
  cracked the pre-splitter rule (possessive single prefix + branch-(f)
  backtrack) via PCRE2/Viktor oracles.
- **Real text from real weights, no torch**: `chat_text` generates
  coherent English from the depth-12 checkpoint; Fortran val-bpb ≈ 0.58
  interim vs torch tag 0.375 (48-row confirmation pending).
- **Refuted along the way:** WRAP trash→gold (format memorization +
  collapse), Muon refinement, pretokenized bin-dataloader, Chinese-literacy
  regime, CS-roleplay-15M. All recorded in HEP with belief updates.

## Project structure

```
src/                — fpm package "fortran_gpt" (the implementation)
  lib/                RMSNorm, matmul, RoPE, causal+GQA attention, ReLU^2,
                      gpt_forward, weight loader, BPE tokenizer tables+encode
  app/                infer (random demo), chat (ids in/out), chat_text
                      (stdin text -> stdout text), eval_bpb, tokdiff
  test/               kernel + wiring tests (fortran-fpm test, 10/10)
  fpm.toml            [dependencies] openmp="*", stdlib="*"
scripts/            — torch-free Python (numpy/tiktoken/pyarrow only)
  export_weights.py   .pt -> .npy via stdlib-only unpickler (bf16-safe)
  chat_driver.py      one-command inference (encode -> chat -> decode)
  eval_driver.py      torch-free val_bpb (pinned shard, BOS packing replica)
  tokdiff_driver.py   Fortran-vs-tiktoken differential test
  export_tokenizer.py one-time BPE/Unicode table export
train.py            — retired Python training record (needs torch; kept as
                      the wiring reference the Fortran engine was verified
                      against). Do not train with it: no working torch left.
prepare.py          — upstream data/tokenizer/metric reference (needs torch;
                      the fixed bpb metric is replicated in eval_driver.py).
checkpoints/        — .pt files (gitignored, local only)
hep/registry.jsonl  — hypothesis audit trail (use hep.py to query it)
flake.nix           — nix dev shell: gfortran + fortran-fpm (torch-free)
```

## Quick start: inference in 3 commands

Requirements: Nix with flakes, ~2 GB free (weights are 373 MB + build).

```bash
# 1. enter the shell (provides gfortran + fortran-fpm)
nix develop

# 2. weights already exported? if not (needs a .pt checkpoint):
/usr/bin/python3 scripts/export_weights.py checkpoints/checkpoint_depth12_step264_0.3750.pt ~/.cache/autoresearch/weights_depth12

# 3. talk to the depth-12 model — zero Python processes:
printf '%s' "Alan Turing theorized that computers" | \
  ./src/build/*/app/chat_text ~/.cache/autoresearch/tok_tables \
    ~/.cache/autoresearch/weights_depth12 20
```

Or the guided version (encode/decode handled):

```bash
nix develop --command bash -c 'export OMP_NUM_THREADS=8;
  .venv-numpy/bin/python3 scripts/chat_driver.py "The capital of France is" --n 20'
```

Run the proofs: `cd src && fortran-fpm test` (kernels),
`/tmp/parity.py` (wiring vs train.py), `scripts/eval_driver.py --rows 4` (val bpb).

## Design choices (how this fork differs)

- **Fortran is the implementation, not a port.** New math goes in
  `src/lib/` with a parity test; Python/Torch is retired, not maintained.
- **Hypothesis-Evolution Protocol (HEP).** Every experiment is registered
  (`python3 hep.py propose`), evidenced, and transitioned — 21 hypotheses,
  beliefs 0.15–0.96. The registry is the lab notebook; read it before
  starting anything.
- **No-torch-chat protocol.** Never reinstall torch to "quickly check"
  something — the torch-free path (numpy harness, stdlib unpickler,
  tiktoken-only drivers) exists precisely so the dependency stays dead.
- **`(b)`-branch discipline.** The BPE pre-splitter rule above is load-bearing;
  touch `tokenizer_encode.f90` only with `tokdiff_driver.py --n 200` green.
- **Small files.** Each Fortran module stays under 300 lines; split before
  growing. Tests live next to the code (`src/test/`), not in notebooks.

## Upstream docs worth keeping

Karpathy's small-compute tuning guide (TinyStories, smaller vocabs/seq-lens,
`WINDOW_PATTERN "L"`, depth as the main knob) still applies to any future
Python-side training. The 5-minute fixed-budget methodology is what produced
the depth-12 checkpoint this fork now runs natively.

trouble with pi harness? export PATH="/home/pauli/.local/share/pi-node/node-v22.23.2-linux-x64/bin:$PATH"

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
