from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BUSI parquet from Hugging Face and export ImageFolder layout.")
    parser.add_argument(
        "--parquet",
        default="data/train-00000-of-00001.parquet",
        help="Parquet filename inside the Hugging Face dataset repository.",
    )
    parser.add_argument("--dataset", default="Angelou0516/BUSI")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/BUSI"))
    return parser.parse_args()


def label_name(value) -> str:
    if isinstance(value, str):
        return value.lower()
    return CLASS_NAMES[int(value)]


def load_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def main() -> None:
    args = parse_args()
    parquet_path = hf_hub_download(
        repo_id=args.dataset,
        repo_type="dataset",
        filename=args.parquet,
        resume_download=True,
    )
    print(f"Using parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    label_column = "label" if "label" in df.columns else "class_label"
    if "image" not in df.columns or label_column not in df.columns:
        raise ValueError(f"Expected image and label/class_label columns, found: {list(df.columns)}")

    counters = {class_name: 0 for class_name in CLASS_NAMES}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        image = load_image(row["image"])
        class_name = label_name(row[label_column])
        if class_name not in counters:
            raise ValueError(f"Unexpected class '{class_name}' at row {idx}. Expected {CLASS_NAMES}.")
        counters[class_name] += 1
        class_dir = args.out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image.save(class_dir / f"{class_name}_{counters[class_name]:04d}.png")

    print(f"Saved BUSI images to {args.out_dir}")
    for class_name, count in counters.items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
