# Configs

Optional YAML configuration for `autoregressive/train/train_c2i_position_order.py`,
passed via `--config path/to/file.yaml`. CLI flags always override values
loaded from the config file. You don't need a config at all — every value
here has an equivalent CLI flag — these just save you from retyping the same
flags across runs.

`--config` takes a single file, so the pieces below are references to copy
from into one experiment YAML, not composable includes:

- **`base/default.yaml`** — every field the training script accepts (model,
  dataset, optimizer, scheduler, logging, paths), with its default value.
- **`gpt_variants/{gpt_b,gpt_l,gpt_xl}.yaml`** — architecture + recommended
  training settings for each model size.
- **`position_order/`** — the three ordering strategies from the paper:
  `random.yaml` (full any-order objective), `raster.yaml` (sanity baseline,
  equivalent to continued raster-scan training), `scheduled_100ep.yaml` (the
  ordering curriculum, linearly interpolating raster → random over 100 epochs).

Example: train GPT-L with the ordering curriculum by merging the relevant
fields from `gpt_variants/gpt_l.yaml` and `position_order/scheduled_100ep.yaml`
into your own `my_run.yaml`, then:

```bash
torchrun --nproc_per_node=4 autoregressive/train/train_c2i_position_order.py \
    --config configs/my_run.yaml \
    --code-path /data/imagenet_codes \
    --cloud-save-path /checkpoints
```
