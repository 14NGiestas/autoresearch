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

## The first port attempt failed, and the gate caught it in one run

The L1 backward was ported into attn_bwd_sgemm, the gate was taught the mode, and
the finite difference check failed at once:

    relu/T:  worst |dL/dq - dq| = 0.238E-03     passes, tolerance 2e-3
    L1:      worst |dL/dq - dq| = 0.542E+00     fails, 271 times the tolerance
    FAIL: dQ, dK and dV match finite differences

That is the gate doing its job, and it cost one run. Without it, this backward
would have gone into a training run and produced a wrong gradient, which is the
class of error that is hardest to see.

The formula is not the suspect. It was re-derived by hand: with P = r/S and
S = sum r, the Jacobian gives

    ds_k = r'_k / S * ( dP_k - sum_i dP_i P_i )

which is exactly what the code computes, with inv = 1/S and rowsum = sum dP P.
The derivation in scripts/relu_l1_math.py agrees.

So the defect is in the port, at a place that was not located in this session.
The first guess was the recomputed sum, because at that point SP already holds P
and the sum would be 1. Fixing that changed 0.549 to 0.542, which is no change,
so that was not the cause.

## To reproduce, in one run

Put the mode back in attn_bwd_sgemm and add call test_attn_bwd_sgemm(relu_l1=.true.)
to test_kernels.f90. Then fpm test fails on dQ, dK and dV. The dQ failure alone
rules out the atomics, because dq is a plain write.

## The state that was left

The tree is back to the last green commit, which has the mode in both forwards
and nothing else. The forwards are correct but unexercised, which is declared.
The plan, the derivation and the verified kernel form are in the repository, and
the failing port is described above with its numbers.

## The forward is proven, so the defect is in the backward

The FD check compares the analytic backward against the finite differences of the
forward, and the forward it differentiates is causal_attn, while the analytic one
is attn_bwd_sgemm. So if the two forwards disagreed, the test would fail with a
perfect backward. That had to be ruled out first.

bench_attn already compares the two forwards and prints the maximum difference.
It was given relu_l1 for one run, and then reverted:

    L1:  max |y_naive - y_sgemm| = 0.209E-05
    /T:  max |y_naive - y_sgemm| = 0.274E-05

The two agree to the same tolerance in both modes, and that tolerance is the
summation order of sgemm. So the L1 forward is correct, in both routines, and the
defect is in the backward.

That narrows the search to one block of about ten lines in attn_bwd_sgemm.

## A process mistake, recorded

The failing port was reverted with git checkout, and the diff was not saved
first. So the exact code that failed is gone, and the next attempt has to rebuild
it from the plan. The isolation above took four minutes; rebuilding the port will
take longer than saving the diff would have.

The rule for the next time: before a revert, keep the diff, for example with git
diff > /tmp/port.patch. A revert is a decision to stop, not a reason to lose the
work that produced the failure.

## The pattern of the failure, and what it points at

The port was rebuilt and the gate was run again. The numbers, with the tolerance
of 2e-3:

    mode   dQ          dK          dV
    /T     0.390E-05   0.566E-05   0.526E-05     all pass
    L1     0.542      0.514       0.229E+01     all fail

All three fail, and by order one, not by a constant factor. So the analytic L1
gradient is structurally wrong rather than mis-scaled.

The sharpest of the three is dV. Its analytic value is P transposed times dY,
with dY given, and the forward with the same P was proven correct at 0.209E-05
against the BLAS forward. So a wrong dV means a wrong P in the backward's own
replay, which contradicts the forward proof. The difference therefore has to be
between the two replays, and the backward's replay is the one that writes into
SP, the caller's scratch, and then reads dSbuf for the mask.

The next instrument is a temporary print inside the L1 branch: sm, inv, rowsum,
dPbuf and dSbuf for the first row of the first batch, compared against the same
quantities computed by hand for the test's known fill. That names the term
instead of guessing at it.

The diff of the failing port is saved at /tmp/port_l1.patch, so the rebuild is
instant next time. That was the process lesson from the previous attempt.

## Found it: a scalar that outlives its row. dQ is fixed.

The instrument found the cause, and it is not the formula.

Printing sm, inv and the row's values at the point where they are PRODUCED, and
not where they are consumed, showed the sums are correct and vary per row:

    PROD ii=1  sm=0.0000  SP(1)=0.0000  SP(2)=0.4093
    PROD ii=2  sm=0.4315  SP(1)=0.0000  SP(2)=0.4315
    PROD ii=3  sm=0.6366  SP(1)=0.0000  SP(2)=0.4559
    PROD ii=4  sm=0.8409  SP(1)=0.0000  SP(2)=0.5484

And the consumer printed sm=0.8409 for ii=1. That is row four's value. So the
inv and sm are SCALARS, computed in the first loop and read in a second loop that
runs after it, and they outlive their row.

That is why /T survives: inv is 1/T, the same for every row. It is why the
softmax survives: it recomputes its rowsum inside the second loop. And it is why
L1 failed: it inherited the last row's inv.

The fix is to recompute the row inside the dS loop, from dSbuf, which holds the
signed score of every row.

    dQ, L1: 0.542 before  ->  0.240E-03 now, which passes

The fact that fixed the reading: the SP slices are disjoint, (ii-1)*TT+jj, so SP
does hold the P of every row. What died was only the scalars.

## What is still wrong

dK and dV still fail in L1, at 0.508 and 2.29, while dQ passes. Both dK and dV
accumulate over the GQA group, and both read SP for P, while dQ reads dSbuf and
is a plain write. In /T all three pass, so the accumulation itself is right.

The next hypothesis is the accumulation path, not the formula and not the
scalars: check what dV's gemm reads for P and what dK's gemm reads for dS, and
whether either reads anything left over from the first loop.

The patch with the dQ fix applied is saved at /tmp/port_l1_v2.patch.

## Three hypotheses, all refuted by measurement, and the clue that is left

Hypothesis one: the formula. Refuted by hand and by scripts/relu_l1_math.py, which
agrees with the code to 1e-17.

Hypothesis two: the degenerate row. The first row has a zero relu sum, where the
forward rule returns a zero row and is therefore discontinuous, so the finite
difference there has no meaning. The test was run with all positive q and k, so
no row is degenerate. dK and dV still failed, at 0.622 and 2.09. Refuted.

Hypothesis three: something clobbers dSbuf between the dQ gemm and the dK gemm.
Two prints were placed, one before each gemm, and they are identical:

    ANTES-dQ  dSbuf(1..3) = -0.1085E-01  0.0000E+00  0.0000E+00
    ANTES-dK  dSbuf(1..3) = -0.1085E-01  0.0000E+00  0.0000E+00

Refuted.

## What that leaves

dQ passes with the L1 fix, at 0.188E-03, and dK fails at 0.622, while both read
the same dSbuf and the prints prove the buffer is identical for both. So the
defect is not in dS and not in the buffer.

The difference between the three consumers is what they do with the result:

    dQ  sgemm('N','N', ...)  plain write, 0.0 beta
    dV  sgemm('N','T', ...)  accumulated,  1.0 beta
    dK  sgemm('N','T', ...)  accumulated,  1.0 beta

The one that passes is the one that writes. The next thing to check is therefore
the accumulation into dkv and the fold that follows it, in the loops that were
never touched by this port and that are the same for every mode. The reason they
pass in relu/T and in softmax, and fail only in L1, still has to be explained, and
that explanation is the missing piece.

The patch with the dQ fix is at /tmp/port_l1_v3.patch.

## Done: the L1 backward in the BLAS path is proven

The finite difference check approves all three components:

    dQ 0.155E-04   dK 0.178E-04   dV 0.820E-05     tolerance 2E-3

Four hypotheses were refuted by measurement, each with its test and its numbers,
and all are in debug/registry.jsonl. The real cause was in none of them: the FD
loops for k and v called the forward without the l1 mode, so they compared the L1
analytic against a relu/T finite difference. dQ passed by accident, because its
loop did carry the flag.

The lesson, and it is the most useful of the day: a gate can be wrong too, and a
wrong gate produces a false negative that looks exactly like a kernel bug. What
unmasked it was refuting every kernel hypothesis by measurement until the only
piece left was the one I had never read, the test itself.

## What is left, and why it stopped here

The naive backward, attn_bwd, does not know the mode yet. Porting it requires
reordering: the S of the L1 depends on the whole row, and there the dcv is built
in a single pass. The test declares the skip, with the reason printed, in the
style of the doc-masked gap.

Running the probe, softmax against relu/T against relu/L1, needs the mode to be
reachable from the model, and that is plumbing that was not done. The chain is
mechanical but it is twelve edits across three files:

  fortran_train.f90   three signatures (relu_attn at 192, 320, 576), each with a
                      declaration, a local, and a pass-through
  fortran_blas path   the two model call sites that reach attn_sgemm and
                      attn_bwd_sgemm, at fortran_train.f90:267 and :417
  train_run.f90       accept relu_l1 in --attn-fn, at the validation on line 147

It stopped because this is the pattern that failed three times in this session: a
run of small edits made late, each of which can leave a half-connected path. The
next attempt does one file, runs the gate, and only then moves to the next.
