# program.md — agent instructions for this fork (Sep 2026)

Upstream `program.md` (5-minute GPU budget, `uv run train.py`, "only edit
train.py") no longer applies. PyTorch is ditched, the model is pure Fortran,
and work is hypothesis-driven. This file is edited by the human; the agent
follows it autonomously (do not ask the user — operate, then report).

## Setup (do once per session)

1. **Shell:** `nix develop` from the repo root (provides `gfortran`,
   `fortran-fpm`; torch-free by design — never create a torch venv).
   Python work uses `.venv-numpy/bin/python3` (numpy/tiktoken/pyarrow only).
2. **Read state:** `python3 hep.py status` (21 hypotheses; `hyp_34ea7c`
   is the main line at 0.96), then `git log --oneline -3` and `git status`.
3. **Verify artifacts:** `~/.cache/autoresearch/` holds `data/` shards,
   `tokenizer/tokenizer.pkl`, `weights_depth12/` (74 `.npy`), `tok_tables/`.
   If weights are missing: `/usr/bin/python3 scripts/export_weights.py
   checkpoints/<ckpt>.pt <out>` (stdlib only — never install torch).
4. **Confirm and go:** summarize state in one paragraph, then proceed.

## Experimentation

Work is registered before it starts:

```bash
python3 hep.py propose --statement "..." --prior 0.5
python3 hep.py evidence --hyp hyp_X --kind test --direction supports \
  --prior 0.5 --updated 0.7 --rationale "..." --source "path:line"
python3 hep.py transition --hyp hyp_X --state under_test  # proposed|under_test|supported|refuted|dormant
```

**What you CAN do:**

- Add math in `src/lib/` (one module per concern, <300 lines/file),
  always with a parity test (`src/test/`, `fortran-fpm test` green).
- Add apps in `src/app/` (`chat_text`, `eval_bpb` patterns to follow).
- Add torch-free drivers in `scripts/` (numpy/tiktoken/pyarrow only).
- Train small models — only via future Fortran training loop work
  (`hyp_34ea7c` direction); Python training is retired (no working torch).

**What you CANNOT do:**

- Reinstall torch, create torch venvs, or "quickly check with torch".
- Modify the fixed metric semantics (bpb on pinned val shard
  `shard_06542`, BOS packing, byte-weighted CE — replicated in
  `scripts/eval_driver.py`).
- Commit checkpoints, logs, `node_modules`, venvs (all gitignored).
- Touch `tokenizer_encode.f90` without `tokdiff_driver.py --n 200` green
  (the `(b)`-branch rule is load-bearing; see README).

## The loop (overnight operation)

This is the adapted core of the initial `program.md`: pick → build →
measure → keep or discard → repeat until morning. HEP replaces
`results.tsv` as the lab notebook (it is append-only; never rewrite it).

1. **Baseline first.** Every run starts from known numbers, not vibes:
   `fortran-fpm test` (10/10), parity err (`/tmp/parity.py`, expect ~2e-7),
   `tokdiff_driver.py --n 200` (exact), depth-12 reference points
   (torch tag 0.375, Fortran interim 0.58). If any baseline is red, fix
   it before experimenting — a broken baseline poisons every verdict.
2. **Pick one hypothesis** (`hep.py status`; lowest-hanging belief first).
   State the falsifiable prediction up front (e.g. "KV-cache keeps bpb
   within 1e-6 while cutting 30-token latency 5×").
3. **Implement small.** One module or one app change, <300 lines/file,
   following `chat_text`/`eval_bpb` patterns. Touch `tokenizer_encode.f90`
   only with the diff green.
4. **Measure with the fixed instruments** — `fortran-fpm test`, parity,
   tokdiff, `eval_driver.py`. Same rows, same flags, or the numbers are
   incomparable. Approx costs: test ~1 min, parity ~1 min, tokdiff-200
   ~2 min, eval Row ~95 s.
5. **Keep or discard (simplicity criterion applies).** Improvement kept
   only with evidence recorded (`hep.py evidence` + `transition`); a
   0.01 gain from 200 lines of hacks is discarded, a 0.0 with deleted
   code is kept. Discards get one evidence line too (why it failed).
6. **Commit green states only** (`git add src scripts ...`, never
   checkpoints/logs/venvs), push the branch, sleep capped at 1800s per
   call with `nohup` for the long evals.

Morning report = HEP delta since last report: hypotheses moved, numbers,
verdicts, disk, next pick. That is the whole job.

**Budgets:** VRAM ceiling is irrelevant now (CPU inference); disk is the
constraint (keep ≥10 GB free — check `df -h /`). `sleep` capped at 1800s
per call; use `nohup ... &` for long runs (`eval_driver --rows 48` ≈ 75 min).
`fortran-fpm` builds cache in `src/build/` (gitignored, safe to delete).

## Environment gotchas (learned the hard way)

- Run `nix develop` from the repo root (its hook used to spawn venvs per
  cwd — neutered, but stay in root anyway). Never export a foreign
  `LD_LIBRARY_PATH` into `nix` itself (breaks its own glibc).
- Inside `nix develop`, the binary is `fortran-fpm`, not `fpm`
  (nixpkgs `fpm` is Ruby's). Use `fortran-fpm run <app> -- args...`
  instead of hunting `build/*/app/` paths.
- Fortran is case-insensitive (`D` ≡ `d`): subroutine args must not
  shadow locals by case. `bind(c)` scalars without `VALUE` arrive
  **by reference** (ctypes: `byref`). Whole-array args to `(*)` dummies
  may pass descriptors — prefer explicit-shape + contiguous sections.
- OpenMP: every private-shared variable goes in `private(...)`; a missing
  one passes all tests at H=1 and explodes at H=4 (see `kb` incident).
- Never `inquire(size=)` on pipes/stdin (gfortran returns 0) — read
  bytes to EOF. A green exit with piped stdout still deserves a look
  at the actual bytes (garbage once rode through `tail` undetected).
- Aborted tool calls reap the whole process group, `nohup` included:
  launch-and-return in one call, check in the next.
- Several `build/<hash>/` dirs is normal fpm behavior (per-target flag
  sets), but globs may grab stale apps — prefer `fortran-fpm run`, or
  resolve newest explicitly and never trust `build/*` blindly.
- fpm 0.10.1 has no `[build] openmp` key — OpenMP comes from the
  `[dependencies] openmp = "*"` metapackage; `stdlib = "*"` gives
  `stdlib_io_npy` (weight loading). `iomsg` args must be deferred-length
  `character(len=:), allocatable`.

## Literature backlog (arXiv survey, Sep 2026 — watchdog-added)

Surveyed for the two live fronts (sampling loops, small-model training).
Status markers: HAVE (in `sample.f90`), NEXT (concrete slice), LATER.

### Decoding / repetition (front: greedy + temp-0.3 loops)

- HAVE: presence/frequency penalties, no-repeat n-gram block, temp + top-p
  (`src/lib/sample.f90`, flags `--pres --freq --nblock --temp --topp`).
- NEXT — CTRL-style repetition penalty (Keskar et al., arXiv:1909.05858):
  multiplicative discount (logits of seen tokens / θ, θ≈1.2), distinct from
  our additive pres/freq. ~15 lines next to `apply_penalties`, flag `--rep`.
- NEXT — windowed (forgetting) penalty (Zhu et al., EMNLP 2023,
  arXiv:2310.14971): penalize only the last W tokens + a length penalty
  against over-short outputs. Explains why full-history pres/freq needs
  retuning per run; flag `--pwin`.
- NEXT — DRY, Don't Repeat Yourself (arXiv:2608.22761): pres/freq/ngram act
  on token recurrence and stop *sequential* loops only at fluency-killing
  strengths; DRY penalizes the loop structure itself. Fits our DOS-loop
  observations exactly; implement after `--rep`.
- LATER — LZ penalty (Ginart et al., arXiv:2504.20131): LZ77-codelength
  penalty; reported to allow greedy decoding with ~0% degeneration where
  freq/repetition penalties still fail (up to 4%). Needs per-step
  match-length computation — feasible in Fortran, but profile first.
- LATER — contrastive search (Su et al., arXiv:2210.14140) / adaptive
  variant (arXiv:2407.18698): needs hidden-state cosine similarity per step.
  Heavier; revisit only if penalty-family stalls. Background reading:
  nucleus sampling + degeneration survey (Holtzman et al., arXiv:1904.09751).

### Small-model training (front: lr/batch choices, hyp_fa2388 line)

- SmolTulu (arXiv:2412.08347, ablations at 135M ≈ our ~100M scale): LR to
  batch-size ratio is task-split — reasoning (ARC/GSM8K) wants *higher*
  ratios, pattern matching (HellaSwag/IFEval) wants lower. Our val-bpb is
  pattern-side; do not blindly raise LR for reasoning hopes. Slice: ratio
  sweep recorded as HEP evidence, not vibes.
- Small-batch Adam rule (Marek et al., NeurIPS 2025, arXiv:2507.07101): hold
  Adam second-moment *half-life fixed in tokens*, not steps (scale β2 with
  batch tokens); small batches then train stably and are *more* robust to
  hyperparameter choice. Our batches are small — apply the β2 rule before
  blaming LR. Also: no gradient accumulation on a single device.
- Warmup (NeurIPS 2024, "Analyzing & reducing the need for LR warmup in
  GPT training"): warmup's job is keeping early update sizes small; verify
  our warmup covers the steps where nll is volatile (steps ~101-110
  showed 0.02→0.72 swings) rather than extending it blindly.

## Output format

Report: what ran (commands), numbers (bpb/parity-err/tokenizer diffs),
theory verdict (supports/refutes/unclear + why), HEP updates made, disk
state, and the next staged step. Keep it tight — files changed, key diffs,
no travelogues.
