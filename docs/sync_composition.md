# sync_composition — the cost of a synchronization

This note records one line of work. The numbers are bpb on a fixed holdout of
100 lines. The model is the 3M model (d96) of this repository, and the binary is
`train_run` with BLAS attention. The init is `/tmp/mix/init3m` and the pool is
`/tmp/mix/rows_f0.npy`, unless the text says otherwise. The driver is
`scripts/fedavg_rounds.py`.

Two corrections are already inside the text below. One corrects the source of the
gain of the outer optimizer. The other corrects the mechanism of the reset.

## The instrument

| item | value |
|---|---|
| workers | K=4, four `train_run` processes |
| synchronization | the average of the weights every tau steps for each worker |
| budget | 1952 steps for each worker, which is 16 rounds at tau=122 |
| compute | K times 1952, which is 7808 worker steps |
| wall-clock reference | one run of 1952 steps: **2.44475** |
| compute reference | one run of 7808 steps: **2.12000** |
| one-shot | one round of 1952 steps: 2.50360, which is worse than no composition |

## What to do with the optimizer state (a table of 3 by 2)

| K=4, tau=122, 1952 steps for each worker | no outer | outer Nesterov, lr=0.7 |
|---|---|---|
| moments **averaged** between workers | 2.69018 | 2.42053 |
| moments **for each worker**, no average | **2.68616** | **2.42092** |
| moments **reset** to zero | **2.49459** | **2.33183** |

**CORRECTED: the poison is stale state, and not the average.** To average the
moments and to keep them for each worker are the same result, at -0.004 and
+0.0004. Both sit inside the noise. The reset wins by 0.19 without the outer and
by 0.09 with it. Our first reading said that the average of the moments between
workers is the poison. That reading is refuted.

## The source of the gain of the outer optimizer (a control at K=1)

| K=1, tau=122, 16 rounds, the same compute | bpb |
|---|---|
| moments carried | 2.71348 |
| plus outer Nesterov at 0.7 | **2.49638** |

**CORRECTED: about 80 percent of the outer gain is a better optimizer, and not a
repair of the synchronization.** Without any composition the outer gives -0.217.
With K=4 it gives -0.270. So the fair comparison is not composition against a
plain single machine. The fair comparison is **composition with the outer
(2.33183) against a single machine with the same outer (2.49638)**. The
composition then wins by 0.16 bpb at the same wall-clock, because both sides use
the same optimizer.

## The cost of the round structure

K=1 with **one round** matches the old reference to five digits, at 2.44476
against 2.44475. So the plumbing is neutral. K=1 with **16 rounds** costs +0.27,
at 2.71348. The outer recovers most of it, and +0.05 remains. A linear warmup of
2 steps restarts at every invocation, and 32 of 1952 steps are warmup. The state
of the optimizer is the other candidate.

| K=1, 1952 steps, moments carried | rounds | bpb |
|---|---|---|
| tau=1952 | 1 | 2.44476 |
| tau=488 | 4 | 2.54990 |
| tau=244 | 8 | 2.62045 |
| tau=122 | 16 | 2.71348 |
| tau=61 | 32 | 2.77547 |
| tau=122, moments reset | 16 | **2.55094** |

The cost grows with the **number of rounds** and not with tau. The reset recovers
about two thirds of it. At K=1 there is no average at all, and the state agrees
with the weights. The reset still wins by 0.16. So the mechanism is not stale
state. The likely mechanism is the one that federated learning already knows:
a restart of Adam at each round lifts the step size again.

The comparison between arms stays valid, because every arm pays this cost. The
comparison against the one-round reference needs the note above.

## The frequency of a sync saturates

| tau | syncs | lr 0.4 | **lr 0.7** | lr 1.0 |
|---|---|---|---|---|
| 488 | 4 | - | 2.55000 | 1.5 explodes |
| 244 | 8 | 2.50783 | 2.34745 | 2.32873 |
| 122 | 16 | - | **2.33183** | - |
| 61 | 32 | 2.38053 | **2.32519** | 2.38800 |

At tau=61 the value 0.7 is a clear local minimum, because both neighbors are
worse. So the flat result is not a bad lr. With the best lr, tau=61 ties tau=122
at -0.007. **A double sync rate buys nothing**, and the cost of the
communication, about 4 KB for one token, is not the limit.

The table also corrects an earlier suspicion. I suspected that tau and lr_outer
are not separable. They are: 0.7 is the best value at every tau in the test.

## K saturates

| K, tau=122, 16 rounds, 1952 steps for each worker | bpb | worker steps | part of the ideal |
|---|---|---|---|
| 4 | 2.33183 | 7808 | 35 percent |
| 8 | **2.32762** | 15616 | 28 percent |

K=8 ties K=4 with **a double compute**, and the difference is small at every
comparable point. **CORRECTED interpretation:** the effective batch for one
*update* does **grow** with K, because one update covers K times tau times T
tokens. The saturation is in the quality for one token, when the run trades
updates for batch. Composition spends updates, and this model needs many. The
reference of 7808 updates beats the reference of 16 updates at the same token
count.

## The path that remains: the step

The worker log gives **124928 tokens in 68.5 s**, which is **1823 tokens per
second** on 8 threads. That is **30 GFLOP/s**, or 3.8 GFLOP/s for one core.
B=1 makes every GEMM small: M=1024, N=K=d=96, which is 0.019 GFLOP for one call.
Composition gave 1.35 times. The step efficiency holds 3 to 5 times, and that is
the larger number.

The measurement of the step is closed now. These values are measured:

| component | result |
|---|---|
| batching, B from 1 to 16 | throughput is flat: 858, 863, and 849 tokens per second |
| elementwise kernels | 13.2 ms, which is 1.1 percent of a step |
| GEMMs | 483 GFLOP/s with sgemm |
| attention | about 50 to 60 percent of a step |

The attention is the bottleneck. The root cause is the arch: `head_dim=16` gives
9.8 GFLOP/s, and `head_dim=128` gives 74.3 GFLOP/s at the same T and the same
FLOPs. The reason is the score matrix of T by T for each head, with a dot product
of width 16.

One confound appeared on the way. Each worker uses about 8 BLAS threads, and the
job asks for 8 cores with 4 workers in parallel. That is oversubscription. The
comparison by tokens for each worker stays valid. The comparison by wall-clock
needs `OMP_NUM_THREADS` for each worker, or a larger allocation. A measurement
with 16 threads is 2.3 times **slower** than one with 8.

## Open items

| item | where |
|---|---|
| the quality cost of head_dim, at fixed kv | job 154 |
| K=2 and K=16 | to schedule |
| 2 or 3 seeds for each arm | to schedule |

## References

* `docs/tier_probe.md` — the other end, memory and energy.
* `hep/` — `hyp_ea9457` (this line), `hyp_8f9016` (the saturation in K), and
  `hyp_c3aed6` (step efficiency).
* `docs/fortran_gpt.md` — the library. `docs/writing.md` — the writing rules.
