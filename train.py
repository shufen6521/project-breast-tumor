from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import CLASS_NAMES, DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.data import prepare_busi_splits
from src.metrics import classification_report, confusion_matrix
from src.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BUSI breast ultrasound classifier.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/BUSI"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--model", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    return (weights / weights.mean()).to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = int(targets.numel())
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return total_loss / max(total_items, 1), all_targets, all_preds


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not (args.data_dir / "train").exists():
        prepare_busi_splits(args.raw_dir, args.data_dir, seed=args.seed)

    train_tfms, eval_tfms = make_transforms(args.image_size)
    train_ds = datasets.ImageFolder(args.data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(args.data_dir / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(args.data_dir / "test", transform=eval_tfms)

    if train_ds.classes != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_path = args.out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_y, train_pred = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_y, val_pred = run_epoch(model, val_loader, criterion, device)
        train_report = classification_report(train_y, train_pred, CLASS_NAMES)
        val_report = classification_report(val_y, val_pred, CLASS_NAMES)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_f1={train_report['macro_f1']:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_report['macro_f1']:.4f}"
        )

        if val_report["macro_f1"] > best_val_f1:
            best_val_f1 = val_report["macro_f1"]
            torch.save(
                {
                    "model_name": args.model,
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                    "state_dict": model.state_dict(),
                    "val_report": val_report,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model = build_model(checkpoint["model_name"], num_classes=len(checkpoint["class_names"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_y, test_pred = run_epoch(model, test_loader, criterion, device)
    test_report = classification_report(test_y, test_pred, CLASS_NAMES)
    test_report["loss"] = test_loss
    test_report["confusion_matrix"] = confusion_matrix(test_y, test_pred, len(CLASS_NAMES))

    metrics = {
        "model_name": args.model,
        "image_size": args.image_size,
        "classes": CLASS_NAMES,
        "validation": checkpoint["val_report"],
        "test": test_report,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {best_path}")
    print(f"saved metrics: {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

