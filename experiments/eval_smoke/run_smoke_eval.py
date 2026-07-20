"""Checkpoint-free smoke test for the linear-probe evaluation harness.

Registers a `voxel_baseline` model: downsampled, brain-masked voxel intensities
used directly as frozen features. It exercises the full harness (dataset build,
transform, batched feature extraction, CV splits, ridge/logistic fit, metrics,
artifact writing) without needing a pretrained sMRI MAE checkpoint, and doubles
as a floor to compare learned representations against.

Usage:
    uv run python experiments/eval_smoke/run_smoke_eval.py dlbs_age --limit 16
"""

import argparse

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evaluation.main_linear import main
from evaluation.models.registry import register_model

# Feature grid per axis. 8^3 = 512-dim features, cheap enough to run on CPU.
GRID = 8


class VoxelBaseline(nn.Module):
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return batch["image"].flatten(1).float()


class VoxelBaselineTransform:
    def __init__(self, grid: int = GRID):
        self.grid = grid

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]:
        # datasets' Nifti1ImageWrapper breaks nibabel's reorientation (its __init__
        # takes only the wrapped image), so rebuild a plain Nifti1Image first.
        img = nib.as_closest_canonical(nib.Nifti1Image.from_image(img))
        # reorientation can flip axes, leaving negative strides that from_numpy rejects
        data = torch.from_numpy(np.ascontiguousarray(img.get_fdata(dtype=np.float32)))

        # z-score over brain voxels, background to zero (matches SmriMaeTransform)
        mask = data > data.mean()
        brain = data[mask]
        mean, std = brain.mean(), brain.std(correction=0).clamp_min(1e-6)
        data = torch.where(mask, (data - mean) / std, 0.0)

        # average-pool to a fixed grid so volumes of any shape give equal-length features
        data = F.adaptive_avg_pool3d(data[None, None], self.grid).squeeze(0)
        return {"image": data}


@register_model
def voxel_baseline(grid: int = GRID):
    return VoxelBaseline(), VoxelBaselineTransform(grid)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, default="dlbs_age", nargs="?")
    parser.add_argument("--limit", type=int, default=16, help="max images to load")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        "voxel_baseline",
        args.task,
        overrides=[
            f"task_kwargs.limit={args.limit}",
            f"task_kwargs.n_splits={args.folds}",
            f"device={args.device}",
            "num_workers=0",
            "batch_size=2",
            "name=smoke__voxel_baseline__" + args.task,
        ],
    )


if __name__ == "__main__":
    cli()
