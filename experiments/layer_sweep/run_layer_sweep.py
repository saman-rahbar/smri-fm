"""Layer-wise linear-probe sweep for the sMRI MAE encoder.

Answers: "which encoder layer gives the best frozen features for this task?"
Three people (Mihir's layer-16 and block-20 probes, Chucksss's readout
argument) have separately observed that the final layer is not the best
one -- this produces the actual curve instead of a spot-check on one or two
layers.

Runs the encoder once per image (via layer_extractor.SmriMaeMultiLayerBackbone),
then fits the same ridge/logistic CV probe used by evaluation.main_linear at
every layer, using identical train/test splits across layers (task.split()
is deterministic given a fixed seed) so differences in score are attributable
to the features, not to different folds.

Usage:
    uv run python experiments/layer_sweep/run_layer_sweep.py \
        --ckpt-path /path/to/checkpoint.pth --task dlbs_age --device cuda
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import smri_mae.model_mae as models_mae
from evaluation.main_linear import ESTIMATORS, TransformDataset, aggregate_folds, set_seed, to_device
from evaluation.models.smri_mae import SmriMaeTransform
from evaluation.tasks.metrics import classification_score
from evaluation.tasks.registry import create_task
from layer_extractor import SmriMaeMultiLayerBackbone

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_multilayer_model(ckpt_path: str, global_pool: str = "patch"):
    """Same loading logic as evaluation.models.smri_mae.smri_mae(), except
    the encoder is wrapped for multi-layer extraction instead of final-layer-
    only pooling."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = ckpt["args"]

    model_fn = models_mae.__dict__[args["model"]]
    model = model_fn(
        img_size=args["img_size"],
        in_chans=args.get("in_chans", 1),
        patch_size=args["patch_size"],
        **(args.get("model_kwargs") or {}),
    )
    model.load_state_dict(ckpt["model"])

    backbone = SmriMaeMultiLayerBackbone(
        model.encoder,
        global_pool=global_pool,
        pad_to_multiple=args.get("pad_to_multiple", 32),
    )
    transform = SmriMaeTransform(img_size=args["img_size"])
    return backbone, transform


@torch.inference_mode()
def extract_all_layers(model, dataset, transform, device, batch_size, num_workers):
    loader = DataLoader(
        TransformDataset(dataset, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()

    features: dict[int, list[torch.Tensor]] = {}
    targets = []
    for i, batch in enumerate(loader):
        targets.extend(batch.pop("target"))
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            layer_embeds = model(batch)
        for layer, embed in layer_embeds.items():
            features.setdefault(layer, []).append(embed.cpu().float())
        if i == 0:
            logger.info(f"layers captured: {sorted(layer_embeds)}")

    X_by_layer = {layer: torch.cat(v).numpy() for layer, v in features.items()}
    y = np.asarray(targets)
    return X_by_layer, y


def run_sweep(task, X_by_layer: dict[int, np.ndarray], y: np.ndarray, seed: int) -> list[dict]:
    fit = ESTIMATORS[task.kind]
    positive_label = getattr(task, "positive_label", None)

    rows = []
    for layer in sorted(X_by_layer):
        X = X_by_layer[layer]
        fold_metrics = []
        # task.split() is deterministic given a fixed random_state, so every
        # layer is scored on the identical train/test folds.
        for train_idx, test_idx in task.split():
            estimator = fit(X[train_idx], y[train_idx], seed)
            pred = estimator.predict(X[test_idx])
            y_score = None
            if task.kind == "classification" and positive_label is not None:
                y_score = classification_score(estimator, X[test_idx], positive_label)
            fold_metrics.append(task.metrics(y[test_idx], pred, test_idx, y_score=y_score))
        summary = aggregate_folds(fold_metrics)
        rows.append({"layer": layer, **summary})
        logger.info(f"layer {layer}: " + " ".join(f"{k}={v:.4f}" for k, v in summary.items()))
    return rows


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--task", default="dlbs_age")
    parser.add_argument("--global-pool", default="patch", choices=["cls", "reg", "patch"])
    parser.add_argument("--limit", type=int, default=None, help="max images to load")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4466)
    parser.add_argument("--output", default="output/layer_sweep")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    task_kwargs = {"n_splits": args.folds}
    if args.limit is not None:
        task_kwargs["limit"] = args.limit
    task = create_task(args.task, default_seed=args.seed, **task_kwargs)
    logger.info(f"task: {args.task} ({task.kind})")

    model, transform = load_multilayer_model(args.ckpt_path, global_pool=args.global_pool)
    model.to(device)
    logger.info(f"checkpoint: {args.ckpt_path}")
    logger.info(f"layers in encoder: {len(model.encoder.blocks)}")

    dataset = task.dataset()
    logger.info(f"dataset: {len(dataset)} samples")
    X_by_layer, y = extract_all_layers(model, dataset, transform, device, args.batch_size, args.num_workers)

    rows = run_sweep(task, X_by_layer, y, args.seed)

    out_dir = Path(args.output) / f"{Path(args.ckpt_path).stem}__{args.task}__{args.global_pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "layer_sweep.csv", index=False)
    logger.info(f"\n{df.to_markdown(index=False, floatfmt='.4f')}")
    logger.info(f"wrote {out_dir / 'layer_sweep.csv'}")

    try:
        plot_sweep(df, task.kind, out_dir / "layer_sweep.png")
        logger.info(f"wrote {out_dir / 'layer_sweep.png'}")
    except ImportError:
        logger.info("matplotlib not available, skipping plot")


def plot_sweep(df: pd.DataFrame, kind: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    metric = "r2" if kind == "regression" else next(c for c in df.columns if not c.endswith("_std") and c != "layer")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(df["layer"], df[metric], yerr=df.get(f"{metric}_std"), marker="o", capsize=3)
    ax.set_xlabel("encoder block index")
    ax.set_ylabel(metric)
    ax.set_title(f"layer-wise probe: {metric} vs block")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    cli()
