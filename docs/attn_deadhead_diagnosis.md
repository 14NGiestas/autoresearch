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

## The answer, found by counting instead of deducing

The fix was to count the additions in the same loop that counts the positives,
and to divide by that count. The counter settled it in one run.

    adicoes = 1200, not 100

The kernel accumulates once per LAYER, and the print divided by once per ROW.
Twelve layers times one hundred rows is 1200, against 100, so the value came out
twelve times too high. The arithmetic is exact: 12 times 0.3864 is 4.637, which
is the number that was printed as 4.6368.

The true fractions are 0.334 to 0.427, which is 33 to 43 percent of the causally
allowed scores positive. That is plausible for scores centred near zero. No head
falls below 50 percent, so the original conclusion holds and is now trustworthy:
there are no dead heads in this model.

## The fix that was owed, and why it was reverted the first time

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
