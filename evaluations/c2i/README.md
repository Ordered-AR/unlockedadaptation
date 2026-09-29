# FID / sFID / IS / Precision / Recall evaluation

Adapted from [OpenAI's guided-diffusion evaluator](https://github.com/openai/guided-diffusion/tree/main/evaluations).
Metrics are computed from batches of samples stored as `.npz` (numpy) files.

## Installation

```
pip install tensorflow numpy scipy
```

If you hit `undefined symbol` errors from `libtfkernel_sobol_op.so`, delete that
file from your TensorFlow install; it's a known packaging issue unrelated to
this evaluator.

## Reference batch

Class-conditional generation is evaluated on ImageNet 256x256 against the
reference batch:
[VIRTUAL_imagenet256_labeled.npz](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)

## Running

Generate a `.npz` of samples with
[sample_c2i_position_order_ddp.py](../../autoregressive/sample/sample_c2i_position_order_ddp.py),
then:

```
python evaluations/c2i/evaluator.py VIRTUAL_imagenet256_labeled.npz samples.npz
```

The script downloads the ~100MB InceptionV3 checkpoint used for activations
into the current working directory on first run. Example output:

```
computing reference batch activations...
computing/reading reference batch statistics...
computing sample batch activations...
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: 215.83
FID: 3.94
sFID: 6.14
Precision: 0.83
Recall: 0.53
```
