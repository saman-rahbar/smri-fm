# Evaluation Smoke Run

A checkpoint-free end-to-end exercise of the linear-probe evaluation harness
(`src/evaluation`). It answers "is the eval plumbing working on this machine?"
without needing a pretrained sMRI MAE checkpoint.

`run_smoke_eval.py` registers a `voxel_baseline` model: brain-masked, z-scored
volumes average-pooled to an 8x8x8 grid (512-dim) and fed straight to the same
RidgeCV / LogisticRegressionCV probe the real models use. It also serves as a
performance floor — a learned representation that does not beat downsampled
voxels is not yet earning its keep.

## Run

```bash
uv run python experiments/eval_smoke/run_smoke_eval.py dlbs_age --limit 120 --folds 5
```

`dlbs_age` streams T1w images from the public OpenNeuro S3 bucket (ds004856),
so no credentials or local data are needed. `--limit` caps how many images are
downloaded; drop it for the full 464-image wave-1 set.

Artifacts land in `output/eval_linear/smoke__voxel_baseline__<task>/`
(`summary.csv`, `metrics.json`, `config.yaml`, `log.txt`).

## Reproducing the real model evaluation

The smoke run does not reproduce sMRI MAE numbers. That additionally requires:

1. **A pretrained checkpoint.** None is published yet; `smri_mae` takes
   `model_kwargs.ckpt_path=<checkpoint.pth>` produced by
   `src/smri_mae/main_pretrain.py`.
2. **A CUDA machine.** `MaskedEncoder` attention goes through
   `torch.nested.nested_tensor_from_jagged`, which needs torch >= 2.5. The repo
   pins torch 2.8.0, which has no macOS x86_64 wheel, so Intel Macs cannot run
   the encoder at all.
3. **Dataset access** for the gated tasks: ADNI tasks read the gated HF dataset
   `medarc/adni-mini`, and FOMO26 tasks need the challenge download.

Then:

```bash
uv run python -m evaluation.main_linear smri_mae dlbs_age \
  --overrides model_kwargs.ckpt_path=/path/to/checkpoint.pth
```
