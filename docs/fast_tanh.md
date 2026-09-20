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

## Where the cap is used

The cap is in all eight attention kernels:

| path | forward | backward |
|---|---|---|
| training, naive | `causal_attn` | `attn_bwd` |
| training, BLAS | `attn_sgemm` | `attn_bwd_sgemm` |
| document mask | `causal_attn_doc` | `attn_bwd_doc` |
| inference | `attn_step`, `attn_chunk` | not needed |

Every routine takes an OPTIONAL `cap` argument with a default of 0. No existing
call site changed. With a cap of 0 the arithmetic is the same as before, because
`fast_softcap(x, 0)` returns x and multiplying by 1.0 is exact.

The BLAS backward needed no new scratch. `dP = dY V^T` does not depend on P, so
it moved before the softmax, and `dSbuf` now holds the capped scores for the
derivative.

## How the gradient was verified

The finite difference test in single precision does NOT decide. With a cap of 2
it gives 3.02e-3 against a threshold of 3.0e-3, and a smaller step makes it
worse (3.32e-3), because the floor is the roundoff of the step and not the
curvature. So the threshold now comes from the measurement.

The sharp test is a cross-check. The naive backward and the BLAS backward are
two independent implementations of the same gradient, so they must agree:

    |bwd blas - bwd naive| = 2.4e-06 with no cap, 1.2e-06 with cap 2

That is the roundoff level of single precision. The capped gradient is right.

## The run flag

`train_run --logit-cap X` sets the cap for training and for the val probes.
A cap of 0 (the default) means off. The value belongs to the run, not to the
arch identity, because it does not change a shape.

Verified on a trained checkpoint, where the scores are large enough to matter.
One step, learning rate 0:

| cap | nll |
|---|---|
| 0 | 4.11882 |
| 2 | 4.67910 |
| 0.1 | 4.76700 |

The value moves the result, so the flag reaches the kernels. The cost is real
and it is expected: a model trained with no cap suffers when a cap appears at
test time. Train with the cap to use the cap.

An untrained checkpoint cannot show this. Its scores are about 1e-3, so even a
cap of 0.001 changes nothing.

## The run record, and the loud failure

A checkpoint records the cap it was trained with. The key is `run.logit_cap` in
`__metadata__`, written at every save, and the card carries `"logit_cap"` too.

On resume, `require_run_cap` compares the recorded value with the flag:

| case | behavior |
|---|---|
| the key is present and differs | stop, with the two values and the file name |
| the key is present and matches | continue |
| the key is absent (an init, or an old checkpoint) | a note, then continue |

The last row is a design correction. The first version failed when the key was
absent, and that blocked the first capped run forever: no checkpoint had the key
yet, so no run could create one. A note keeps the fact visible without the dead
end.

The comparison is on the cap only, and the tolerance is 1e-6 relative. A resume
with a different cap changes the model, so it must not pass in silence. That is
the bug class the arch identity already guards for shapes.

Verified end to end on a real checkpoint:

* train one step from a trained checkpoint with `--logit-cap 2`: the note
  appears, the run proceeds, and the file records `run.logit_cap = 2.0`.
* resume that file with `--logit-cap 0`: `FATAL: logit soft cap mismatch`, with
  the recorded value, the asked value, and the file name.
* resume it with `--logit-cap 2`: the run proceeds and the nll is the same
  (4.67910), so the path is deterministic.

The cap does not enter the arch identity. The identity describes the layout of
the weights, and the cap changes no shape. It changes behavior, which is why it
gets its own recorded field.
