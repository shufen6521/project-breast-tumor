from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.config import IMAGENET_MEAN, IMAGENET_STD
from src.models import build_model


def make_preprocess(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_checkpoint_model(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model = build_model(checkpoint["model_name"], num_classes=len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    meta = {
        "model_name": checkpoint["model_name"],
        "class_names": checkpoint["class_names"],
        "image_size": checkpoint["image_size"],
    }
    return model, meta, make_preprocess(checkpoint["image_size"])


@torch.inference_mode()
def predict_image(model, image: Image.Image, preprocess, device: torch.device):
    model.eval()
    tensor = preprocess(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_idx = int(probs.argmax())
    return probs, pred_idx
