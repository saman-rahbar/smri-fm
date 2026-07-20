# Layer-wise Probe Sweep

Answers one question: **which encoder layer gives the best frozen features?**

Right now the eval harness always takes the encoder's *final* layer and
*averages* all its patch tokens into one vector. Three people independently
found that's probably not the best choice:

- Mihir's June layer probes: checkpoint-60 *layer 16* beat the final layer on
  9/14 ADNI tasks.
- Mihir's July "encoder-tail specialization": ViT-H *block 20* beat the final
  layer, with no rank collapse.
- Chucksss's readout argument: averaging tokens also throws away *where* in
  the brain a signal is, which matters for spatially-patterned things like
  tau (see Braak staging).

Nobody has actually produced the full curve — every observation so far is a
spot-check on one or two layers on one model. This script produces it.

## What it does, in plain terms

Normally, extracting features means: run the image through the whole
encoder, take what comes out the other end. This script instead taps the
output after *every* transformer block in a single pass — like measuring the
temperature at every floor of a building on the way up, instead of only at
the roof. Then it trains and scores the exact same linear probe
(RidgeCV/LogisticRegressionCV, the same code `evaluation.main_linear` uses)
separately at each floor, using the *same* train/test split every time so the
comparison is fair — only the features change, not the data split.

The result is one row per layer: how good is brain-age (or whatever task)
prediction if you stop listening to the model at that depth?

## Run

```bash
uv run python experiments/layer_sweep/run_layer_sweep.py \
  --ckpt-path /path/to/checkpoint.pth \
  --task dlbs_age \
  --device cuda
```

`--task` accepts any registered task (`dlbs_age`, `dlbs_sex`, ADNI tasks once
access is approved). `--limit` caps how many images to load, useful for a
fast sanity check before a full run. `--global-pool` switches between
`patch` (default, mean-pool), `cls`, and `reg` token pooling.

Output lands in `output/layer_sweep/<ckpt_name>__<task>__<pool>/`:
`layer_sweep.csv` (one row per layer), `layer_sweep.png` (metric vs. layer).

## Caveat: not yet run end to end

This was written and reasoned through carefully against the real encoder
code (`src/smri_mae/model_mae.py`, `src/smri_mae/modules.py`), and the
pooling logic in `layer_extractor.py` is copied line-for-line from the
existing, working `SmriMaeBackbone.forward()` to minimize risk of a
mismatch. But it has **not been executed** — the encoder's attention path
needs torch >= 2.5, which has no macOS x86_64 wheel, so it cannot be tested
on an Intel Mac.

**Before running a full sweep, smoke-test with `--limit 8 --folds 2`** on
whatever checkpoint you have, and check that `layer_sweep.csv` has one row
per encoder block with sane-looking (not NaN, not identical-across-layers)
metrics. If anything errors, the most likely failure point is the mask
bookkeeping in `SmriMaeMultiLayerBackbone.forward` (`layer_extractor.py`) --
specifically the assumption that `encoder.cat_token_mask(token_mask, B)`
called a second time from outside the encoder exactly reproduces the mask
used internally to pack tokens before the block loop.

## One known difference from the standard model

`layer == depth - 1` (the last block) is **not** identical to what the
standard `smri_mae` eval model returns for the same checkpoint. The standard
model applies `encoder.norm()` (the final LayerNorm) before pooling; this
script pools every layer's *pre-norm* output, since norm is a parameter
trained specifically for the final layer, not a generic per-layer
normalizer. `layer_sweep.csv` should therefore show the last row skewing
slightly worse than the standard model's own number on the same checkpoint
and task -- that gap is expected and is the norm's contribution, not a bug.
