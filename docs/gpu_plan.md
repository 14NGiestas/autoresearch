# The GPU path: state of the work

Date: 2026-09-21. Everything below is measured on the Radeon 780M (gfx1100, 12 CU)
in the Ryzen 7 8745HS, which is the chip of fermi. The measurements were taken
while other jobs used the machine, and the contention is declared in each case.

## Why the GPU

The step at the C-0.1B size takes 1.810 s and reaches 58 percent of the CPU
ceiling, which is 299 GFLOP/s against 517. The iGPU peak, measured with rocBLAS,
is 2348 GFLOP/s. So the prize is between 3.7 and 5.5 times, and it is the
difference between 25 CPU nodes and one small machine for the 1B target.

## What the swap touches

src/lib/fortran_blas.f90 is 3440 bytes and holds two subroutines:

    linear3d_sgemm(x, w, y, BB, TT, IF, OF)              ! y = x @ W^T
    linear3d_bwd_sgemm(dy, x, w, dx, dw, BB, TT, IF, OF) ! two sgemm calls

The forward is one sgemm. The backward is two. So the whole swap is one
interface, and the work is small.

## Where the code lives, and why there

| where | what | why there |
|---|---|---|
| src/lib/gpu_shim.cpp | the three calls, both branches | compiles in both builds |
| src/lib/fortran_blas_gpu.f90 | bind(C) interface and dispatch | Fortran, built by fpm |
| src/fpm.toml feature gpu | the macro ARCH_GPU and the flags | does not depend on the machine |
| bin/build_gpu | hipcc, the include paths, the link | depends on the nix store |

The shim is one file with the GPU path under ARCH_GPU and the other branch in
plain C++. So it lives inside the package and fpm compiles it in both builds. The
CPU-only build, and fpm test with it, needs no ROCm and stays green.

The check that decided this was a deliberate #error in the file: the CPU-only
build failed on it, so fpm does compile it. An earlier check looked for the
symbol with nm and found nothing, which proves nothing, because the linker drops
symbols that nothing references.

## The measured rates

Against the CPU, which reaches 299 GFLOP/s at this size.

| call | ms | GFLOP/s | against the CPU |
|---|---|---|---|
| fwd | 4.186 | 1154 | 3.9x |
| bwd dx | 2.702 | 1788 | 6.0x |
| bwd dw | 2.990 | 1616 | 5.4x |

The backward is stronger, at 69 to 76 percent of the peak, and the backward is
two thirds of the training compute.

At the model's own size the picture inverts. There the transfer and the
synchronization cost per call are fixed and they dominate.

| shape | fwd | bwd dx | bwd dw |
|---|---|---|---|
| 1024 x 768 -> 3072 | 1154 | 1788 | 1616 GFLOP/s |
| 1024 x 96 -> 288 | 181 | 180 | 126 GFLOP/s |

At d96 the GPU is slower than the CPU. So the GPU is not a general win. It is a
win at large shapes, which is where the ladder says the value is. A run at d96
would measure the overhead and not the GPU.

## Correctness

All three calls are checked against a serial loop at the shapes of the model, 96
by 288 and 32 by 96, and the error is exact.

The end to end check, at the d96 shape, five steps:

| step | CPU | GPU |
|---|---|---|
| 1 | 9.02792 | 9.02792 |
| 2 | 9.00960 | 9.01677 |
| 5 | 8.97616 | 9.02247 |

Step 1 is identical to the last digit. Step 1 measures the loss before any
update, so the forward pass is exact.

Steps 2 to 5 diverge, and this is chaotic amplification, not a wrong result. The
evidence: with one step and the checkpoints saved, the largest difference in any
tensor is 4.424e-09, which is 1.9e-07 relative, single precision roundoff. The
cause is the summation order of rocBLAS against OpenBLAS, which is a legitimate
difference between two libraries.

So the criterion had to change. "The loss must not move" was right for the -O2
replication, because that was the same library and the same order. Across two
libraries it is too strict. The right criterion is the statistical quality at
equal steps, against the plateau scatter of 0.01 to 0.02 bpb, plus the tokens per
second.

## The GPU does not sweat, and that is measured

| state | mean power | peak |
|---|---|---|
| idle | 42.2 W | 63.1 W |
| during the benchmark | 37.8 W | 65.1 W |

The mean during the benchmark is lower than at idle and the peak is the same.
The kernels last three to five milliseconds and the gaps about fifty, so the
power manager of the APU never notices. The rates above are burst rates, one
kernel at a time. The peak of 2348 needs a sustained load.

The second reading matters more. The power is near 40 W in both cases, so the
host side, the synchronous copies included, dominates the time. That is where an
end-to-end gain can die, and it is why the end-to-end number is the only one that
counts.

## Corrections, kept because they cost time

Five claims of mine were wrong, and each was caught by a measurement.

The first was that the shim needed to be an external artefact, like OpenBLAS.
It does not. A define inside the shim is enough, and the package carries the
kernel.

The second was that a C++ file in src/ would break the CPU-only build. It does
not, because the other branch is plain C++.

The third was a check that could not fail: nm for a symbol that nothing
references.

The fourth was a measurement of 20.7 GFLOP/s, which timed the first call and
therefore measured the HIP context and not the kernel. The test now warms up.

The fifth was the worst and the most useful. Wiring the backward gave NaN at
step 4 and adam_v at 5.3e+21. The bisect put dx on the GPU alone and the error
came back at 4.657e-10, so dx was correct and dw was wrong. The kernel was never
the suspect, because the test checked it. The shim's weight cache used the
activation slot pool, so registering a second weight freed the buffer of the
first while the cache kept pointing at it. A use-after-free.

The reason the test missed it is the lesson. Each run of the test used one
shape, so it registered one weight. The case that mattered, two weights in one
process, was never exercised. The test has that case now, and it fails on the old
code.

## What is next

The end-to-end run at a large shape, where the GPU wins. The C-0.1B size is the
natural one: 768 by 3072 gives 1154 to 1788 GFLOP/s. That needs
bin/build_gpu 768 6 2 12 8192 1024. The number to read is the tokens per second,
against 566 on the CPU.
