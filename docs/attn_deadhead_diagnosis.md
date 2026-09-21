# The dead-head counter: diagnosis, and the fix that is owed

Date: 2026-09-21. The instrument is eval_bpb --attn-stats, which reports the
fraction of attention scores that are positive, per head.

## What it shows

The fractions come out between 4.01 and 5.12. Values above one are impossible
for a fraction of causal pairs. So the number is wrong, and the print already
declared it: "o divisor esta ERRADO (o valor sai ~4,6x alto)".

The declared warning was right. I read it backwards at first and went looking
for a divisor that was too large.

## Where it is not

The formula in the kernel is right. npos is reset once, before the row loop, so
it sums the causal matrix of one (batch, head) pair. The denominator is
T*(T+1)/2, which is exactly the count of causal pairs. Both are correct.

The print divides by ncontrib, which it reports as 100. So the accumulated value
is about 460, which means one call contributes about 4.6, which is more than one.
The contradiction is in the loop structure, not in the arithmetic.

## The fix that is owed, and why it was not shipped

The correct fix is to stop presuming the denominator and count the additions
inside the same loop that counts the positives. Then the ratio is right by
construction, whatever the nesting is. That is a small change, and it touches
three files: fortran_attn.f90, fortran_gpt.f90 and eval_bpb.f90.

It was attempted, it built wrong, and it was reverted. The reason to revert is
the rule of the house: a half-plumbed counter is worse than a declared bug,
because a declared bug is visible and a half one is not. The tree is clean and
verified, and the fix is the first task of whoever picks this up.

The lesson, and it is the one this project keeps teaching: the value above one
was the whole diagnosis. A fraction that cannot exist said more than any amount
of reading the loops.
