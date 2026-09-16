"""Datasets and file/label utilities for the preprocessed OASIS data."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = "/home/groups/comp3710/OASIS"
SPLITS = ("train", "validate", "test")


def split_directories(data_root, split):
    """Return MRI and segmentation directories for one OASIS split."""
    root = Path(data_root)
    return (
        root / f"keras_png_slices_{split}",
        root / f"keras_png_slices_seg_{split}",
    )


def _image_files(directory):
    return sorted(Path(directory).glob("*.png"))


def paired_paths(image_dir, mask_dir):
    """Pair case_XXX... MRI files with seg_XXX... mask files by filename."""
    pairs = []
    for image_path in _image_files(image_dir):
        mask_name = image_path.name.replace("case_", "seg_", 1)
        mask_path = Path(mask_dir) / mask_name
        if not mask_path.is_file():
            raise FileNotFoundError(f"No mask for {image_path.name}: {mask_path}")
        pairs.append((image_path, mask_path))
    if not pairs:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")
    return pairs


def discover_label_values(mask_dir):
    """Find sorted original pixel values in the training masks only."""
    values = set()
    for path in _image_files(mask_dir):
        values.update(np.unique(np.asarray(Image.open(path))).tolist())
    if not values:
        raise FileNotFoundError(f"No training masks found in {mask_dir}")
    labels = tuple(sorted(int(value) for value in values))
    print(f"Discovered original training labels: {list(labels)}")
    print(f"Number of classes: {len(labels)}")
    return labels


def label_mapping(label_values):
    return {value: index for index, value in enumerate(label_values)}


def _load_image(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _load_mask(path, mapping):
    mask = np.asarray(Image.open(path))
    unknown = set(np.unique(mask).tolist()) - set(mapping)
    if unknown:
        raise ValueError(f"Mask {path} contains labels not in training mapping: {unknown}")
    indexed = np.zeros(mask.shape, dtype=np.int64)
    for original_value, class_index in mapping.items():
        indexed[mask == original_value] = class_index
    return indexed


class OASISImageDataset(Dataset):
    """Grayscale MRI dataset returning tensors in [1, H, W] and [0, 1]."""

    def __init__(self, image_dir):
        self.image_paths = _image_files(image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = torch.from_numpy(_load_image(self.image_paths[index])).unsqueeze(0)
        return image


class OASISSegmentationDataset(Dataset):
    """Paired MRI/mask dataset using a mapping discovered from train masks."""

    def __init__(self, image_dir, mask_dir, mapping):
        self.pairs = paired_paths(image_dir, mask_dir)
        self.mapping = mapping

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = torch.from_numpy(_load_image(image_path)).unsqueeze(0)
        mask = torch.from_numpy(_load_mask(mask_path, self.mapping))
        return image, mask