# Estimate before you measure

Write the expected size of an effect before you queue the job.

## Why

An order of magnitude decides whether the job is worth the queue. It also makes
the error of our own prediction a datum. Without a prediction written down
first, a measurement only confirms what we already believe. When the
measurement disagrees with the prediction, that disagreement is the finding.

## The four fields

Every prediction carries these, and `scripts/effect_estimate.py` holds them:

1. **model** - the arithmetic, in words, so anyone can redo it.
2. **prediction** - the expected number, with its unit.
3. **falsification** - what the measurement must show for the prediction to be
   wrong. A prediction without this is a wish.
4. **measurement** - the value, once it exists. Empty means the job is still in
   the queue.

The script computes what it can and prints the prediction error when the
measurement arrives. A laboratory that does not measure the error of its own
predictions does not know if it is thinking or guessing.

## The worked case

The fast tanh in the attention soft cap. The model counts the calls: one per
causal score pair, so `B*H*L*T*(T+1)/2`, which is 37.8 million calls per step at
d96. At 4 cycles per call the polynomial costs 0.05 s of a 1.211 s step. At 40
cycles the intrinsic tanh costs 0.50 s, which is 42 percent of the step.

| regime | prediction |
|---|---|
| cap off (every run we have) | 0 calls, so exactly 0 percent |
| cap on, polynomial | about 4 percent of the step |
| cap on, intrinsic tanh | about 42 percent of the step |

The window moves with the square root of the step time, so the intrinsic tanh
would cost 1.37 times the time for the same model, or 15 percent of the model
for the same time.

Falsification: if the measured gap between the two binaries is under 0.05 s, the
prediction is wrong by ten times and the fast tanh is not a lever.

Measurement: job 157, still in the queue.

## What this is not

This is not a substitute for measuring. It is a reason to measure the right
thing, and a way to know afterwards whether we understood the system or got
lucky.
