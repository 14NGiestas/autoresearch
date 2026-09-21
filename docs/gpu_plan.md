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

## Step one: done, and it works

gpu/rocblas_shim.c exposes a C ABI over rocBLAS for Fortran. It avoids hipfort,
which is absent, and it avoids the implicit-interface trap of fortran_blas.f90,
because bind(C) declares every type.

The design that matters: the weights stay resident on the device. The 1B has
352 MB of weights, and a copy per step would kill the gain. So the shim
registers a weight once and keeps the device pointer in a cache. The activations
are small and travel on every call.

Build, and the measured command is in gpu/gpubench.c:

    hipcc -O3 -D__HIP_PLATFORM_AMD__ -I$HIP_PATH/include -I$(dirname $RB)/include \
      -o /tmp/shim_test gpu/rocblas_shim.c gpu/shim_test.c -lrocblas -L$RB

Correctness: the maximum error against a serial reference is 0.000e+00, at every
shape.

| shape (BT, IF, OF) | ms per call | GFLOP/s |
|---|---|---|
| 1024, 768, 3072 (MLP up) | 4.812 | 1004 |
| 1024, 768, 768 (MLP down) | 1.969 | 613 |
| 1024, 768, 8192 (head) | 9.988 | 1290 |
| 6, 128, 1024 (QK^T) | 0.088 | 17.9 |

The CPU reaches 299 GFLOP/s at this size. The shim reaches 613 to 1290, with
pageable transfers and a malloc and free on every call still inside. So it lands
in the predicted band.

The small shape confirms the attention decision by a second route: 17.9 GFLOP/s
is dominated by the transfer, so the attention is not worth porting.

The first measurement of this shim read 20.7 GFLOP/s, because the test timed the
first call. The first call pays the HIP context, so that number measured the
context and not the kernel. The test now warms up. The lesson is the lesson of
the day: a number without its conditions is not a number.

## What is next

Wire the shim behind the two subroutines of fortran_blas.f90, behind a run flag,
in the style of --attn-fn. Then the end-to-end run, same seed, same rows, and
compare the loss and the tokens per second.

## Step one, finished: the three calls, correct and fast

The shim compiles as C++ and publishes a C ABI with extern "C". That matters
because hip_runtime.h is C++ only, while rocBLAS alone compiles as C. The test
stays in C.

All three calls are checked against a serial loop on a small shape, with the
standard of the house, an exact error:

    fwd:     0 of 20 wrong
    bwd dx:  verified
    bwd dw:  verified

The build has zero warnings and zero errors.

| call | ms | GFLOP/s | against the CPU 299 |
|---|---|---|---|
| fwd | 4.186 | 1154 | 3.9x |
| bwd dx | 2.702 | 1788 | 6.0x |
| bwd dw | 2.990 | 1616 | 5.4x |

The backward is where the GPU is strongest, at 69 to 76 percent of the 2348
peak, and the backward is two thirds of the training compute. So the effective
kernel ratio is about five, above the 3.7 of the forward alone.

Two defects found by the way, both mine, both recorded.

The first is a design bug: the buffer pool returned the first buffer that fitted,
so an input and an output could share it. The fix is explicit slots.

The second is a lesson about editing: a regex patch mangled this file, and I
rewrote it instead of patching it further. That is the same class as the jobs
migration earlier on the same day, where the pattern matched and the meaning
broke. Twice is enough to make it a rule: rewrite, do not patch.

## What is next

The Fortran bind(C) wrapper over these three calls, behind a run flag in the
style of --attn-fn. Then the end-to-end run, same seed and same rows, comparing
the loss and the tokens per second.
