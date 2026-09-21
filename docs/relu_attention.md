# ReLU attention: measured verdict

Date: 2026-09-21. Instrument: job 6202 (softmax) and 6203 (relu) on halfbeast.
Recipe: d96 h6 kv2 l12, init3m, rows_f0h, lr 6e-4, batch 1, attn blas,
nsteps 2929, nval 100, val_every 100. One arm per job, 4 threads each.

## The question

Job 164 gave relu 4.154340 against softmax 2.352489. Two readings were possible.
The first reading is a different floor: the curve falls at the same slope and
stops higher. The second reading is a collapse: the curve goes flat early,
because the attention died.

## The answer

Both readings are wrong. The gap grows.

| step | softmax | relu | gap |
|---|---|---|---|
| 100 | 3.60165 | 3.60018 | -0.0015 |
| 500 | 2.89304 | 2.89059 | -0.0025 |
| 600 | 2.81996 | 2.82011 | +0.0002 |
| 900 | 2.70798 | 2.71965 | +0.0117 |
| 1200 | 2.62162 | 2.67501 | +0.0534 |
| 1500 | 2.53668 | 2.62358 | +0.0869 |
| 1900 | 2.45821 | 2.58911 | +0.1309 |
| 2300 | 2.40807 | 2.56407 | +0.1560 |

The gap is not constant, so the relu does not sit on a different floor. The
curve of the relu still falls, so the attention does not collapse. The relu
learns at a lower rate, and the deficit accumulates. Half of the deficit
appears after step 1500.

The relu reached 2929 steps with 2.52544.

## The old number

Job 164 reported 4.154340 for the relu. This run does not reproduce it. Job 164
ran at 15:23, before the fix of the thread race in eval_bpb. That race gave
3.577647 at OMP=2, 4.208167 at OMP=4, 4.470370 at OMP=8 and 4.661770 at OMP=16.
The old number falls inside that band. This is a hypothesis with evidence, not
a measured fact.

## Speed

The relu finished 2929 steps while the softmax, under the same contention, was
killed at 2300 steps by the 2-hour limit. The relu is faster. The measurement is
dirty, because the two jobs shared the machine. A clean measurement needs a
dedicated run.

## What the next step needs

The softmax arm needs a longer wall-clock limit. Two hours does not hold 2929
steps at 4 threads. Use three hours, or more threads.
