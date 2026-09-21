# The relu/sum(relu) variant: derivation done, port prepared

Date: 2026-09-21. This is the variant that was left untested, and the one the
literature points at: the critical component of softmax is the L1
normalization, not the exponential.

## Why it is not one line

With P = relu(s) / sum(relu(s)), the denominator depends on every score of the
row, so the backward carries the quotient rule and has two terms:

    ds_k = r'_k * [ dP_k / S  -  (sum_i r_i dP_i) / S^2 ]

The second term is the denominator term. Omitting it is not a small error.

## The derivation is verified, in the repository

scripts/relu_l1_math.py checks it by central differences before any Fortran is
touched.

| case | full backward | without the denominator term |
|---|---|---|
| 1 | 7.2e-12 | 4.9e-03 |
| 3 | 3.2e-10 | 5.5e-01 |
| 5 | 6.2e-11 | 4.9e-02 |

The full form is exact to 1e-10. Dropping the denominator term gives errors from
5e-3 to 5e-1, which is the same class of error that the FD check caught for
relu/T, when the forward divided by the causal length and the backward by T, and
dq came out wrong by 1.2.

## Where the port goes

Two forward sites in src/lib/fortran_attn.f90, one in causal_attn and one in
attn_sgemm. The relu branch computes inv = 1/T today. For the new mode it must
divide by the sum of the relu, which the row loop already walks, so the sum is
one extra accumulator in a loop that exists.

The backward is not in that file. It is inlined in src/lib/fortran_train.f90,
where the attention backward replays the forward. Both places must change
together, and the FD check is what proves they agree.

## The gate: found, and it is in the repository

The FD check was not lost. It is src/test/test_kernels.f90, and it is part of
fpm test. The record named it: the commits 0ffd9ed and 6ac7308, "o porteiro pegou
dois bugs reais" and "o porteiro pegou um bug real", both touch fortran_attn.f90
and test_kernels.f90.

It already exercises the relu path:

    call test_attn_bwd_sgemm(relu_attn=.true.)
    call test_attn_bwd(relu_attn=.true.)

and it uses the standard trick, finite differences of L = <dy, y>. The tolerance
is stated honestly in the file: the sgemm summation order makes it exact only
before BLAS, so the check is 1e-6.

So the acceptance criterion of the port is fpm test, and the gate is ready. The
two bugs the porter found for relu/T were the forward dividing by the causal
length against the backward dividing by T, which made dq wrong by 1.2, and the P
recomputation in attn_bwd_sgemm left unconverted, which made dv wrong by 1.72.

## Why this stopped here

The derivation is the part that must be right, and it is done and verified. The
port is a multi-file change with a check to locate first. It was 05:45, after a
long session, and the same session had already produced a half-plumbed counter
that had to be reverted. Doing a second one at that hour, on a backward pass,
is how a wrong gradient gets committed. The plan is written and the next session
starts with it.
