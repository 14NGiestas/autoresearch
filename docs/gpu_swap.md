# The GPU swap: the unit is the BLAS layer, not the attention block

Date: 2026-09-21. Instrument: bench_attn_split, run on halfbeast with 4 threads
while job 6211 used the machine. The lesson of the day applies: the numbers below
come from a contended machine, and the contention is declared.

## The measurement

The shape of the C-0.1B probe: T=1024, 6 heads, 2 KV heads, head_dim 128.

| head_dim | QK^T | softmax+mask | PV | sgemm/softmax |
|---|---|---|---|---|
| 16 (d96) | 1.750 ms | 2.750 ms (49%) | 1.100 ms | 1.04 |
| 36 (d216) | 3.300 | 2.700 (30%) | 3.000 | 2.33 |
| 60 (d360) | 2.700 | 2.950 (37%) | 2.300 | 1.69 |
| 128 (88M) | 4.650 | 2.800 (24%) | 4.250 | 3.18 |

The softmax plus mask is constant, 2.7 to 2.95 ms at every head_dim, because it
is T-squared work and does not depend on head_dim. The GEMMs grow 3.1 times. So
at the 88M shape the softmax falls from 49 percent to 24 percent of the
attention.

## The step

The measured step at 88M is 1.810 s.

| part | time | share of the step |
|---|---|---|
| forward attention | 140.4 ms | 7.8% |
| GEMMs inside it | 106.8 ms | 5.9% |
| softmax plus mask | 33.6 ms | 1.9% |

By FLOPs per token: attention 43 percent, MLP 51 percent, head 6 percent.

## What this decides

The unit of the swap is the BLAS layer, not the attention block. Porting only the
attention gives 1.07 times, because the whole attention is 7.8 percent of the
step.

The worry written in the header of bench_attn_split says that if the softmax
stays on the CPU, the ceiling of the gain is about 1.8 times. The measurement
resolves that worry: the softmax is 1.9 percent of the step, so it can stay on
the CPU.

The prize, the projections and the MLP, is already an sgemm call today. The swap
is a binding change in fortran_blas.f90, not a rewrite.

## The prediction

The iGPU kernel measured 2137 GFLOP/s at k=128, which is 7.1 times the 299
GFLOP/s that the CPU reaches at this size.

| GEMM share | Amdahl | tokens/s | nodes for 6 months |
|---|---|---|---|
| 85% | 3.72x | 2105 | 0.60 |
| 90% | 4.43x | 2505 | 0.51 |
| 95% | 5.47x | 3094 | 0.41 |

So one iGPU does 1B at 20 tokens per parameter in three to four and a half
months, against 12.7 years on one CPU node.

## What is not measured

This is the forward pass. The backward also calls BLAS, so the share should
hold, but the split of the backward was not measured.

The 7.1 times comes from a pure sgemm microbench, with no memory traffic and no
host synchronization.

The probe at 88M ran with 8 threads, while the CPU ceiling was measured in
another configuration. The efficiency of 58 percent is therefore understated,
and the ratio of 7.1 shrinks.
