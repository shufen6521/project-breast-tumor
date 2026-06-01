from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models import gradcam_target_layer


def make_heatmap(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.8 * cam, 0.0, 1.0)
    green = np.clip(1.8 * (1.0 - np.abs(cam - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(1.4 * (1.0 - cam), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def make_gradcam_overlay(
    model,
    image: Image.Image,
    preprocess,
    target_class: int,
    device: torch.device,
    model_name: str,
    alpha: float = 0.42,
) -> Image.Image:
    model.eval()
    activations = []
    gradients = []
    target_layer = gradcam_target_layer(model, model_name)

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor = preprocess(image).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        score = logits[0, target_class]
        score.backward()

        acts = activations[-1]
        grads = gradients[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.size[::-1], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / max(cam_np.max() - cam_np.min(), 1e-8)

        base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        heat = make_heatmap(cam_np)
        overlay = np.clip((1.0 - alpha) * base + alpha * heat, 0.0, 1.0)
        return Image.fromarray((overlay * 255).astype(np.uint8))
    finally:
        forward_handle.remove()
        backward_handle.remove()

