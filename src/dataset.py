"""Dataset utilities for the Kermany chest X-ray one-class protocol.

Paper Section 2.1: the training split contains 1,341 radiologist-validated
*normal* chest X-rays only (one-class classification, OCC). The test split mixes
normal images with "pneumonia" images, which are the pathological anomalies.
No training image appears in the test split.

This module intentionally does **not** download anything. Point ``root`` at a
local copy of the dataset; see the README for the download instructions.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

#: Number of normal training radiographs used by the paper.
TRAIN_NORMAL_COUNT = 1341


@dataclass
class Sample:
    """One radiograph and its OCC label (1 = abnormal, 0 = normal)."""

    path: str
    label: int
    split: str


def discover_manifest(
    root: str,
) -> Tuple[List[Sample], List[Sample]]:
    """Build ``(train_normal, test_mixed)`` sample lists from a dataset root.

    The Kermany release is organised as ``train/<class>`` and ``test/<class>``
    with classes ``NORMAL`` and ``PNEUMONIA``. Directory names are matched
    case-insensitively, and the ``test_A`` / ``test_B`` naming used by some
    redistributions is accepted as well.

    Returns
    -------
    ``(train_samples, test_samples)`` where ``train_samples`` are all labelled
    normal and ``test_samples`` carry both classes.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"dataset root {root!r} does not exist; see README section "
            "'Dataset' for download instructions"
        )

    train_samples: List[Sample] = []
    test_samples: List[Sample] = []

    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root).replace("\\", "/").lower()
        if rel in {".", ""}:
            continue
        in_train = "train" in rel.split("/")
        in_test = "test" in rel.split("/")
        if not (in_train or in_test):
            continue
        is_normal = "normal" in rel
        is_pneumonia = "pneumonia" in rel
        if not (is_normal or is_pneumonia):
            continue

        for name in filenames:
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue
            path = os.path.join(dirpath, name)
            if in_train and is_normal:
                train_samples.append(Sample(path, 0, "train"))
            elif in_test:
                test_samples.append(Sample(path, 1 if is_pneumonia else 0, "test"))

    return train_samples, test_samples
def load_manifest_csv(path: str) -> List[Sample]:
    """Read a manifest produced by :func:`write_manifest_csv`."""
    samples: List[Sample] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            samples.append(
                Sample(row["path"], int(row["label"]), row.get("split", "test"))
            )
    return samples


def write_manifest_csv(path: str, samples: Sequence[Sample]) -> None:
    """Persist a sample list so splits can be frozen and shared."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "split"])
        for sample in samples:
            writer.writerow([sample.path, sample.label, sample.split])


class ChestXrayOCCDataset(Dataset):
    """Minimal ``Dataset`` returning raw [0, 255] CHW tensors plus labels.

    Images are returned **before** normalization so the anatomical prior filter
    can threshold on the physical [0, 255] intensity scale; call
    :func:`src.streams.normalize_batch` downstream.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image
        import numpy as np

        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("L")  # chest radiographs are single channel
            array = np.asarray(image, dtype="float32")
        tensor = torch.from_numpy(array).unsqueeze(0)  # (1, H, W)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, sample.label, sample.path


def split_normal_train(
    samples: Sequence[Sample], val_ratio: float = 0.1, seed: int = 0
) -> Tuple[List[Sample], List[Sample]]:
    """Hold out a normal-only validation split for hyper-parameter selection.

    The paper tunes ``lambda`` and the decision threshold on a validation set
    (Section 2.3 / Section 3.1), so this split mimics that protocol.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must lie in [0, 1)")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(samples), generator=generator).tolist()
    n_val = int(round(len(samples) * val_ratio))
    val_idx = set(order[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val
