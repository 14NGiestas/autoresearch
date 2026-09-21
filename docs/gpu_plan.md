# GPU plan: one interface, one kernel, and a measured target

Date: 2026-09-21. This plan follows the measurement in docs/gpu_swap.md and the
decision to keep the CPU path.

## What the swap touches

src/lib/fortran_blas.f90 is 3440 bytes and holds two subroutines:

    linear3d_sgemm(x, w, y, BB, TT, IF, OF)          ! y = x @ W^T
    linear3d_bwd_sgemm(dy, x, w, dx, dw, BB, TT, IF, OF)

The forward is one sgemm. The backward is two. So the whole swap is one
interface, and the work is small.

## Why not a library binding first

The header of that file records the trap: nixpkgs OpenBLAS is ILP64, and an
LP64 interface reads stack garbage. The mfi package was tried and reverted for
exactly that. A library binding carries an ABI question. A kernel of our own
carries none.

But rocBLAS is available and already runs: gpu/gpubench.c links it and measured
the numbers below.

## The shapes, at the C-0.1B size

| call | m | n | k |
|---|---|---|---|
| QKV | 2304 | 1024 | 768 |
| MLP up | 3072 | 1024 | 768 |
| MLP down | 768 | 1024 | 3072 |
| head | 8192 | 1024 | 768 |
| attention QK^T | 1024 | 6 | 128 |

## The measured target

| shape | rocBLAS | percent of peak 2348 |
|---|---|---|
| k=128 | 2137 GFLOP/s | 91% |
| MLP up | 1264 GFLOP/s | 54% |
| k=96 | 1060 GFLOP/s | 45% |

The library is at 91 percent on the small k and at 54 percent on the dominant
shape. So the effective gain of the GPU is 4.3 to 5.3 times, not 7.1. That lands
on the Amdahl band of docs/gpu_swap.md, so the ladder does not move: one iGPU
does 1B at 20 tokens per parameter in three to four and a half months.

And it also gives the specialized kernel a measured target. The library leaves
1.7 times on the table on the shape that matters most.

## The order of work

First, rocBLAS behind the two subroutines. It is proven to run, the ABI question
is already answered in gpubench.c, and it gives the end-to-end number, which is
the only number that counts.

Second, the specialized kernel, and only if the end-to-end gain falls short of
the prediction. The shapes are compile-time constants, because the arch is a
build feature. A kernel can bake them in. A general library cannot.

## The success criterion

The loss must not move. The -O2 replication set the standard: same bits per
byte, difference 5e-6. A GPU path that changes the loss is worthless.

The speed must beat the CPU path in the same run, same seed, same rows.

## Keep both

The CPU path stays runnable. It is the reference for correctness, and the
comparison between the two is the measurement.
