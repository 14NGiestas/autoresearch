# fast_tanh and the attention soft cap

The module `src/lib/fortran_math.f90` holds two functions.

## fast_tanh

The coefficients come from fastGPT (certik/fastGPT, `gpt2.f90`). The polynomial
is odd in x, of degree 17, and it is valid on [-5, 5]. Outside that range the
function returns +1 or -1, because tanh saturates there.

**Measured error, not a hope.** `test_math.f90` reports two numbers over the
range [-10, 10] in steps of 0.005:

| value | result |
|---|---|
| largest absolute error against the exact tanh | **3.05e-3** at x = -4.91 |
| largest error weighted by the derivative | **2.73e-3** |

The second number is the one that matters. The tanh enters a softmax, so an
error in a region with a small derivative reaches the result with less weight.
My first guess was that the error sits only where the tanh saturates, and the
measurement says no: the weighted error is almost the same as the plain one.
The error is spread out.

For comparison, the cheap Pade form `x*(27 + x*x)/(27 + 9*x*x)` has a worst case
error near 0.025, which is eight times worse.

## The soft cap

The attention soft cap is `cap * tanh(x / cap)`. Gemma 2 and Grok-1 use it, and
`fast_softcap` implements it with `fast_tanh`. A cap of 0 or less means no cap.

Two rules come from the field and they are not optional:

1. **The cap goes before the mask.** A masked position must not reach the
   softmax. A cap applied after the mask leaves a small value on a masked
   position, and information from a future token then leaks into a past token.
2. **The forward and the backward must call the same function.** The gradient
   test then proves that the pair agrees. It does not prove that the value equals
   the exact tanh. That is a separate measurement, and it is the table above.

The derivative is cheap. The backward already replays the scores, so it can keep
the capped value `s_capped` and use

    d/dx [cap * tanh(x/cap)] = 1 - (s_capped/cap)^2

No second call to tanh is needed.

## What remains

The cap is not implemented in the attention kernels yet. The work is precise:

* `causal_attn` and `attn_sgemm`: apply `fast_softcap(score, cap)` before the
  mask. Both paths.
* `attn_bwd` and `attn_bwd_sgemm`: keep the capped score in a small array, and
  multiply the gradient of the score by `1 - (s_capped/cap)^2`. Both paths.
* An optional argument `cap` in all four routines, with a default of 0. That
  keeps every existing call site unchanged.
* A finite-difference test with `cap > 0`, in `test_kernels.f90`. This is the
  acceptance test. A new gradient without it is a belief.
* A runtime flag `--logit-cap` in `train_run`, recorded in the checkpoint card as
  `run.logit_cap`, and a loud failure when a resumed checkpoint was trained with
  a different cap. The arch identity stays out of this: it describes the layout
  of the weights, and the cap does not change a shape.
