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
- fpm 0.10.1 has no `[build] openmp` key — OpenMP comes from the
  `[dependencies] openmp = "*"` metapackage; `stdlib = "*"` gives
  `stdlib_io_npy` (weight loading). `iomsg` args must be deferred-length
  `character(len=:), allocatable`.

## Output format

Report: what ran (commands), numbers (bpb/parity-err/tokenizer diffs),
theory verdict (supports/refutes/unclear + why), HEP updates made, disk
state, and the next staged step. Keep it tight — files changed, key diffs,
no travelogues.
