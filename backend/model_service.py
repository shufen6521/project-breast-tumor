from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from src.gradcam import make_gradcam_overlay
from src.inference import load_checkpoint_model, predict_image


CHECKPOINT_PATH = Path("models/best_model.pt")
METRICS_PATH = Path("models/metrics.json")


class ModelNotReadyError(RuntimeError):
    """Raised when the trained checkpoint is not available."""


class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot be decoded as an image."""


class TumorClassifierService:
    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        metrics_path: Path = METRICS_PATH,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.metrics_path = metrics_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._meta: dict | None = None
        self._preprocess = None

    @property
    def checkpoint_exists(self) -> bool:
        return self.checkpoint_path.exists()

    @property
    def metrics_exists(self) -> bool:
        return self.metrics_path.exists()

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.exists():
            raise ModelNotReadyError(f"Missing checkpoint: {self.checkpoint_path}")

        model, meta, preprocess = load_checkpoint_model(self.checkpoint_path)
        model.to(self.device)
        model.eval()
        self._model = model
        self._meta = meta
        self._preprocess = preprocess

    def read_metrics(self) -> dict:
        if not self.metrics_path.exists():
            return {"available": False}
        return {"available": True, **json.loads(self.metrics_path.read_text(encoding="utf-8"))}

    def predict(self, image_bytes: bytes) -> dict:
        self.load()
        assert self._model is not None
        assert self._meta is not None
        assert self._preprocess is not None

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("Uploaded file is not a readable image.") from exc

        probs, pred_idx = predict_image(self._model, image, self._preprocess, self.device)
        class_names = self._meta["class_names"]
        predicted_class = class_names[pred_idx]

        overlay = make_gradcam_overlay(
            self._model,
            image,
            self._preprocess,
            pred_idx,
            self.device,
            self._meta["model_name"],
        )

        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        encoded_overlay = base64.b64encode(buffer.getvalue()).decode("ascii")

        return {
            "predicted_class": predicted_class,
            "predicted_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": [
                {"class_name": class_name, "probability": float(probs[idx])}
                for idx, class_name in enumerate(class_names)
            ],
            "gradcam_image": f"data:image/png;base64,{encoded_overlay}",
            "model_name": self._meta["model_name"],
            "image_size": int(self._meta["image_size"]),
            "device": str(self.device),
        }


service = TumorClassifierService()
