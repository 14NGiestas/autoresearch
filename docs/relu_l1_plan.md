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

## The gate, widened, and the limit of the doc-masked path

The gate already covered the relu path in two calls. It now covers two more
routines, and both are green.

test_causal_attn gained the two branches in its serial reference, which is an
independent implementation and therefore a stronger check than a cross-check. It
runs twice now, plain and with relu, and the relu path passes.

test_attn_sgemm gained cap and relu_attn, passed to both the naive and the BLAS
call, for the GQA and the MHA cases. The maximum error in the relu run is 0 and
1.1e-7. This matters because a layout mistake in the relu path is exactly where
the dv error hid before, and that case is now guarded.

Two routines of the attention family do not take relu_attn at all:
causal_attn_doc and attn_bwd_doc. Their signatures are
"... docstart, cap)". So the document-masked path does not implement relu.

That is not a silent fallback. The argument does not exist, so a caller cannot
ask for relu on that path, and the mistake shows up at compile time rather than
as a wrong number. The limit is real and it is declared here: the relu variant
has no document-masked implementation.

If the relu family survives the l1 test, the doc path needs it too, and the two
routines are the places to add it.

## The kernel form, verified, and smaller than the plan predicted

The plan gave the backward as

    ds_k = r'_k * [ dP_k / S  -  (sum_i r_i dP_i) / S^2 ]

Reading the relu/T backward in attn_bwd showed a useful fact. It already
computes ssum = sum_i dp_i P_i, and since r_i = P_i S, the numerator of the
second term is S times ssum. So the whole thing collapses to

    ds_k = r'_k / S * ( dP_k - ssum )

One subtraction, and ssum already exists. The kernel form was checked against
the derivative and against finite differences:

    |kernel - derivative| = 1e-17     |kernel - FD| = 1e-11

which is exact, with the FD noise being the 1e-11.

## So the port is three lines per backward site

The relu branch of attn_bwd and of attn_bwd_sgemm, and the inlined copy in
fortran_train.f90, each need:

1. dcv(ss) = merge(1/S, 0, sc(ss) > 0)      instead of merge(1/T, 0, ...)
2. P = relu(S)/S                            instead of relu(S)/T
3. ds = dcv(ss) * (dpv(ss) - ssum)          instead of dpv(ss)*dcv(ss)

plus the sum S over the causal row, which the replay loop already walks. The
zero case, S equal to zero, gives a zero row rather than a NaN.

The gate must gain the mode in the same commit, because the tree has to be green
at the end of the task, and a forward that knows L1 with a backward that does not
is exactly the mismatch the FD check exists to catch.
