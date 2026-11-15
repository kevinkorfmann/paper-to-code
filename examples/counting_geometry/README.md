# Saturated count coordinates

Source: Shaffy et al., *Counting with Attention and LayerNorm: Norm Scaling
and Representation Geometry*, [October 12, 2026 preprint](https://arxiv.org/abs/2610.16653v1), Section 5.
This original implementation isolates its scalar segment/prototype construction:
t(n)=n/(n+gamma); class c scores 2*t(c)*x-t(c)^2. Bounded coefficients do not
prevent shrinking class margins. We evaluate equivalent squared distances to
avoid subtracting nearly equal logits. This is not a training reproduction or
verification of the paper's norm theorems.

## Run

CPython 3.14.5, standard library only; deterministic, no random seed needed.
From the repository root:

```sh
python -m unittest examples.counting_geometry.test_core -v
python -m examples.counting_geometry.demo
python -m examples.counting_geometry.demo > /tmp/counting-output.json
diff -u examples/counting_geometry/sample_output.json /tmp/counting-output.json
```

Our supplied gamma=2 experiment represents n identical target tokens and one
boundary with weight 2 and value zero. No distractors in the main experiment.
It checks 17, 65 and 257 count classes. All unquantized binary64 coordinates
classify correctly here. Quantizing only the input coordinate to multiples of
1/256 reduces correct counts to 17, 36 and 42, respectively. Counts 255 and
256 collide at 0.9921875 and both predict 254. Prototype coefficients are NOT
quantized. This grid is not IEEE low-precision emulation; 257 grid points
include both endpoints, so “8 bits” describes fractional resolution, not total
storage width. The classifier cannot output an unrepresented class: count 300
predicts 256.

A second negative control supplies distractors with equal attention weight
and zero target value. Inputs (n,m)=(1,0) and (2,2) share coordinate 1/3.
The scalar coordinate alone then loses information. This does not contradict
the paper's full vector classification construction for distractor inputs.

A separate explicit regressor gamma*t(n), with gamma=8192, stays within 0.5
of counts through 64, but fails that tolerance by 128. It is not minimax,
and gamma is a supplied score ratio, not a trainable-vector norm budget.

Tests use independent exact rational logits, exact rational quantization,
explicit token-weight summation, spacing identities, decision boundaries,
invalid inputs and CLI/sample replay. Results are component-level numerical
checks, not speed, precision robustness at arbitrary lengths, learned gains,
LayerNorm experiments or language-model performance. No upstream code copied.
