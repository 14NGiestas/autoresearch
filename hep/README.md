# HEP — Hypothesis Evolution Protocol (applied to `autoresearch`)

This directory implements the **autonomous from-scratch pretraining research loop**
described by Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch),
structured as an auditable **Hypothesis Evolution Protocol** per:

> Takahara & Mizoguchi, *"Toward Auditable AI Scientists: A Hypothesis Evolution
> Protocol for LLM Agents"*, arXiv:2607.09195v1 (2026).
> https://arxiv.org/html/2607.09195v1

## Migração para o protocolo JS (2026-09-17)

A registry era escrita pelo `hep.py` (Python), agora aposentado: a implementação
de referência é `packages/hep-js` (npm `hep-protocol`, arXiv:2607.09195v1),
invocada por `bin/hep` (build local, offline) com `npx hep-protocol` como reserva.

A migração foi forçada por dois defeitos que o `hep.py` permitia e que o
`verify` da referência detecta:

1. **`seq` duplicado (96)**: dois processos leram o mesmo tail e anexaram, então a
   cadeia tinha dois elos com o mesmo número (226 registros, 225 `seq` únicos).
2. **`kind` fora do enum**: 101 registros usavam valores que o protocolo não
   aceita (`test`, `empirical`, `measurement`, `note`, `external`, `supports`).

`scripts/rechain_registry.mjs` fez a reparação: mapeou os `kind` inválidos para o
enum (`test|empirical|measurement|supports|inconclusive` → `experiment`,
`external` → `literature`, `note|refine` → `analysis`), renumerou `seq` na ordem
real de append e recalculou `prev`/`hash` com a canonicalização da referência.
**Nenhum payload foi removido, reordenado ou reescrito além do campo `kind`** —
216 dos 226 elos mudaram apenas por causa da renumeração/cadeia. O original está
no histórico do git; `bin/hep verify` agora responde *hash chain valid*.

## The mapping

The paper's HEP is an agent harness for scientific discovery. We apply its
**auditability spine** — hypotheses as persistent, hash-chained objects with an
explicit belief and lifecycle — to the autoresearch experiment loop:

| HEP concept (paper)            | Here (autoresearch)                                 |
|-------------------------------|-----------------------------------------------------|
| Hypothesis (a scientific claim) | A training experiment = a change to `train.py`    |
| Prior belief P(H)              | Expected val_bpb improvement before the run          |
| Test                           | Run `train.py` (fixed time budget) in `tmux`        |
| Evidence                       | Observed `val_bpb` from `run.log`                   |
| Belief update                  | New P(H) from the measured delta vs. parent         |
| Verdict / lifecycle transition | `keep` (supported) / `discard` (refuted) / `dormant`|
| Generation mechanism           | `de-novo` / `refine` (tweak parent) / `merge` / `inspired-by` |

Every event is appended to `registry.jsonl` (append-only, **hash-chained** to the
previous event, like the paper's event-sourced HEP Registry), so each hypothesis
carries an unbroken, verifiable record from proposal to verdict.

## Usage

```bash
bin/hep propose   --statement "..." --prior 0.5 --mechanism refine --parents hyp_XXX
bin/hep evidence  --hyp hyp_XXX --kind simulation --direction supports \
                         --prior 0.5 --updated 0.55 --bpb 1.23 --commit <sha> --rationale "..."
bin/hep transition --hyp hyp_XXX --state supported     # or refuted / dormant
bin/hep status
```

## Training data

We train **from scratch on our own hoard corpus** (`hoard_multi`), not Karpathy's
climbmix. `make_hoard_data.py` renders the structured agent sessions
(opencode + pi; claude/gemini/chatgpt pending normalization) into parquet `text`
shards consumed by `prepare.py`, and retrains the BPE tokenizer on our data
(~65M tokens, 88.7k documents). The model therefore learns the agentic
conversation / tool-use distribution of the harvested sessions.

## Run mechanics

- One GPU (AMD Radeon, gfx1100, 16 GB, ROCm). `nix develop` provides the env.
- Each experiment is launched in a `tmux` session:
  `./bin/tmux new-session -d -s train "nix develop --command python3 train.py > run.log 2>&1"`
- Metric: `val_bpb` (bits per byte), lower is better, vocab-size-independent.
- Results also logged to `results.tsv` (untracked) per program.md.
