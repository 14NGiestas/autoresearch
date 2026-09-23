# fortran_gpt — the library

This package is a GPT stack in pure Fortran for a CPU. It holds no Python and no
torch in any path of execution. It holds the forward pass, the backward pass, the
training loop, the KV cache and generation, safetensors checkpoints, and energy,
CPU, and IO measured by the process itself.

The package is `src/`, with `src/fpm.toml` and fpm 0.13.

## Modules in `src/lib/`

| module | what it does |
|---|---|
| `fortran_arch.f90` | **the arch in one place**: D_MODEL, N_HEAD, N_KV, HD, N_LAYER, VV, TT, BOS, plus `write_arch_txt`, `read_arch_txt`, `require_arch`, `check_shape`, `arch_canonical`, and `arch_id` |
| `fortran_kinds.f90` | `wp = real32`, and one line changes it to real64 |
| `fortran_blas.f90` | the BLAS interface. The OpenBLAS of the nix store is **ILP64** |
| `fortran_linear.f90` | `linear3d` and `wte_lookup` |
| `fortran_rmsnorm.f90`, `fortran_rope.f90` | RMSNorm and RoPE |
| `fortran_attn.f90` | causal attention, naive and with `sgemm` |
| `fortran_gpt.f90` | the forward pass |
| `fortran_backward.f90` | the backward pass |
| `fortran_train.f90` | one training step, and the validation bpb |
| `fortran_adamw.f90`, `fortran_adam_state.f90` | AdamW and the optimizer state |
| `fortran_muon.f90` | Muon, with Newton-Schulz orthogonalization |
| `fortran_qkhop.f90` | QK-hop, for the routing experiments |
| `fortran_kv.f90` | the KV cache and `gpt_step`, for incremental decode |
| `fortran_spec.f90` | speculative decode with a lookup drafter |
| `fortran_recurrent.f90` | the recurrent passes, through `--loops` |
| `sample.f90` | sampling: temperature, top-k, and penalties |
| `fortran_data.f90` | the rows and the batches. It checks the header of an npy file |
| `fortran_chat.f90` | the chat template |
| `fortran_sys.f90` | `mkdir_p`, `dir_exists`, and `exit` |
| `load_weights.f90` | safetensors checkpoints: weights, optimizer state, arch metadata, the energy card, the lineage, and `read_arch_any` |
| `fortran_probe.f90` | **the machine probe**: bandwidth, latency, and joules for each memory level. It knows nothing about a model |
| `fortran_texture.f90` | **the texture of a text, over bytes**: alpha, byte_alto, palavra_plausivel, distinct 1 to 3, loop and repetition, plus the three-step scale |
| `tokenizer_tables.f90`, `tokenizer_encode.f90` | BPE in pure Fortran. `scripts/export_tokenizer.py` writes the tables |

## Apps in `src/app/`

| app | what it does |
|---|---|
| `train_run.f90` | the real training: steps, validation, checkpoints, and the energy trace |
| `train_1step.f90`, `train_loop.f90` | smaller steps for a test |
| `eval_bpb.f90`, `bpb_agg.f90` | the bpb on a holdout, and its aggregation |
| `infer.f90`, `repl.f90`, `chat_text.f90` | generation: batch, REPL, and chat |
| `merge_ckpt.f90` | the merge of checkpoints, with or without selection |
| `tier_probe.f90` | recompute against read of a KV window. Two modes: infer and train |
| `arch_id.f90` | the identity of an arch: the canonical string and the id |
| `texture_scan.f90` | the texture of one file, with a JSON line |
| `bench_attn.f90`, `bench_batch.f90`, `bench_gemm.f90`, `bench_elem.f90`, `spec_bench.f90` | kernel benchmarks |
| `tokdiff.f90` | the difference between two tokenizations |

## Tests in `src/test/`

| test | what it holds |
|---|---|
| `test_kernels.f90` | each kernel against an inline reference, and the gradients against finite differences |
| `test_st_ckpt.f90` | the round trip of a safetensors checkpoint, and the metadata guards |
| `test_energy_tier.f90` | the invariants of the energy probe: the page cache does not touch the disk, `FADV_DONTNEED` works, the warm bandwidth is at least half the cold one, and the latency for one IO is at least twice the sequential cost |
| `test_arch_id.f90` | the identity of an arch: the same input gives the same id, one changed field gives a new id, and the known configurations do not collide |
| `test_texture.f90` | the texture panel: prose beats a salad on the important values, the scale is correct, and an empty text does not stop the program |

```bash
nix develop .#cpu-only --command bash -c "cd src && fortran-fpm test"
nix develop .#cpu-only --command bash -c "cd src && fortran-fpm test test_texture"
```

## Build and gate

```bash
bin/fbuild              # a normal build
bin/fstrict             # the strict gate. It applies -Werror, and only to us
bin/fbuild --werror     # sends the work to bin/fstrict
```

`bin/fstrict` builds everything with the default flags, and then rebuilds **our
sources only** with `-Werror`. The reason is the flag path. The `--flag` of fpm
also reaches the dependencies, and stdlib alone reports `compare-reals` 104
times, `conversion` 93 times, and `unused-dummy-argument` 20 times. A global
`-Werror` would then need a long list of exceptions. That list would weaken the
gate for the code that matters.

Use `--build-dir /tmp/...` to test without a change to a build tree that a
running job uses. A job resolves its binary at the moment of the run.

The dependencies are `stdlib` from the registry, `openmp`, `safetensors` from git
at tag `v0.1.3`, `M_CLI2` from git at the commit `0704ed3`, and `fortran_energy`
by path.

## The arch: build any configuration without an edit

The arch lives in `fortran_arch.f90` as `#define ARCH_*`. The file holds the
canonical configuration of the repository. To build another arch, pass the
defines. Do not edit the file.

```bash
bin/build_arch 96 6 2 12 8192 1024 /tmp/b96
```

The script uses `--features` when the arch is declared in `src/fpm.toml`, and it
falls back to `--flag -D...` when it is not. The binary then reports its own
identity:

```
bin/build_arch 96 6 2 12 8192 1024 /tmp/b96
  ...
  id ca674ae5302a6cc9
```

That id is the same as the id of the checkpoints of the running experiments. The
binary and the checkpoint agree, and a name of a directory no longer carries that
information.

`bin/arch_check.sh` compares the Fortran id against the Python id for the same
checkpoint. The two must agree, because two implementations of one rule need a
test.

## The identity of an arch

The arch was in three places: the `parameter` values of `fortran_arch.f90`, the
`__metadata__` of the safetensors file, and the `arch.txt` sidecar. The link
between a binary and a checkpoint was a **name of a directory**. That fault put
the source tree at d216 while the running experiments used d96.

The package now holds these parts:

| part | what it does |
|---|---|
| `arch_canonical()` | one canonical string with a fixed order and no space |
| `arch_id()` | 16 hex digits: the identity for a selection and a check |
| `read_arch_any()` | one reader: the `__metadata__` first, and `arch.txt` after |
| `check_arch_selfconsistent()` | the document must agree with itself: the fields, the canonical string, and the id |
| `scripts/arch.py` | the same rule in Python, with a command line |
| `bin/arch_check.sh` | the check of the agreement between Fortran and Python |

The id is not a cryptographic hash. It is a rotation and an exclusive or over the
bytes of the canonical string. That choice has two reasons. It does not depend on
the overflow of a signed integer, which the standard does not define. And it has
the same short implementation in both languages, which makes the agreement test
possible. The id only needs no collision between configurations, and
`test_arch_id.f90` checks that.

Example: the d96 of the experiments gives `ca674ae5302a6cc9`. The source tree at
d216 gives `bec469fb7c561d2c`.

## The texture of a model

`fortran_texture_mod` measures a text **over bytes**. One character is one byte,
and `iachar` gives the byte value. A decode step before the measurement replaces
a bad byte with a replacement character, and that changes the result. The app
`texture_scan.f90` applies the panel to one file.

```bash
./build/*/app/texture_scan SAMPLE.txt --json T --key texture
```

The scale has three steps. Step 1 is the byte floor, where the text is a salad.
Step 2 is a usable generator, with correct words and no loop. Step 3 is a
reasoner, where the model corrects itself. The R1 "aha" moment is step 3.

The test holds the calibration. Real prose gives `palavra_plausivel` near 0.95
and `byte_alto` near 0.005. A trained 3M model gives 0.20 and 0.146, which is
step 1. So the first usable step has not arrived yet, and we can now detect the
moment it does.

The JSON line is the same line that goes into the `__metadata__` of a checkpoint.
The integration is open: the generator still lives inside `app/repl.f90`, which is
a program. A module and an app that sample a checkpoint and write the texture
into the card are the next step.

## Conventions that prevent a fault

1. **The arch is one source.** The shapes live in `fortran_arch_mod` as derived
   values. A new size is `scripts/set_arch.sh`, or a define at the build.
2. **A wrong shape stops the program.** `require_arch` and `check_shape` exist
   because of a real case of silent garbage from a binary and a checkpoint of
   different archs.
3. **A checkpoint carries more than weights.** It carries `arch.*`,
   `energy.J`, `energy.self_measured`, and `lineage.parent`.
4. **Energy for each phase.** Call `energy_mark('name', tokens=n)` in the hot
   path. The trace goes to `energy_trace.csv`, and the card of a checkpoint
   carries the delta.
5. **JSON goes through the package.** Use `json_num` and `json_escape` from
   `safetensors_json`. A NaN becomes `null`, because JSON holds no NaN, and a
   card with NaN is not readable.
6. `wp` is `real32`. BLAS is ILP64. Read `fortran_blas.f90` before a change.

## Where to go next

* `docs/tier_probe.md` — recompute against read, with numbers.
* `docs/sync_composition.md` — the cost of a synchronization.
* `docs/mpi_ddp.md` and `docs/halfbeast.md` — several machines and the queue.
* `docs/writing.md` — the writing rules for this repository.
* Open work: `fortran_energy` is still a path dependency, and this repository has
  no CI, because it depends on the machines of the lab.
