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

## The cost against the benefit, stated safely

The relu costs 0.156 bpb at 2300 steps. The relu is faster, because it finished
2929 steps while the softmax was killed at 2300 under the same contention. The
speed ratio is dirty, and this run does not measure it.

A claim about equal wall-clock needs two premises. The first premise is that the
speed ratio holds. The second premise is that the gap grows without limit. This
run does not show either one.

The safe claim is this. At this scale, and at this length, the relu does not
pay. The extrapolation to long runs stays open. The literature suggests the gap
may stabilize, because the curve of the relu still falls.

## The normalization of this model

This model uses RMSNorm, not LayerNorm. The literature that reports a relu win
uses models with no normalization at all. This model sits in a third cell, so
the clean claims of that literature do not cover this measurement.

The attention here uses the divide-by-T convention, which is the convention of
the Wortsman result in vision. The relu still loses in this language model.

## What the literature separates

Wortsman, the Softplus paper and FLARE replace the softmax in the attention.
The NYU paper and ReLU's Revival replace the MLP activation, which is GELU
against ReLU. Those are different comparisons. Only the first group bears on
this measurement.

The Softplus ablation reports a loss gap of +0.049. FLARE reports that training
a relu from scratch is 59 percent slower. Both readings match the growing gap
measured here.

No paper in this survey reports bpb for relu against softmax in a language
model. That survey is the work of the reviewer, not of this run.

## What the next step needs

The softmax arm needs a longer wall-clock limit. Two hours does not hold 2929
steps at 4 threads. Use three hours, or more threads.

The clean question for the size axis is a d360 with the head_dim controlled. Job
6204 runs exactly that: the same recipe, the same data, the same seed and the
same init template, with ten heads instead of six. The head_dim becomes 36,
which is the head_dim of the d216 point. Only then does the size axis get a
clean answer.
