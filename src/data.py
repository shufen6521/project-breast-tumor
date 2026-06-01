from __future__ import annotations

import random
import shutil
from pathlib import Path

from src.config import CLASS_NAMES


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def is_training_image(path: Path) -> bool:
    name = path.stem.lower()
    return path.suffix.lower() in IMAGE_EXTENSIONS and "mask" not in name


def split_items(items: list[Path], seed: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def prepare_busi_splits(raw_dir: Path, out_dir: Path, seed: int = 42) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"BUSI raw directory not found: {raw_dir}. "
            "Expected data/raw/BUSI/{benign,malignant,normal}/."
        )

    for class_name in CLASS_NAMES:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        images = [p for p in class_dir.iterdir() if p.is_file() and is_training_image(p)]
        if len(images) < 3:
            raise ValueError(f"Need at least 3 images for class '{class_name}', found {len(images)}")

        for split_name, split_paths in split_items(images, seed).items():
            target_dir = out_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in split_paths:
                shutil.copy2(src, target_dir / src.name)

