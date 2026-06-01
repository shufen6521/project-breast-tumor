from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.config import CLASS_NAMES
from src.gradcam import make_gradcam_overlay
from src.inference import load_checkpoint_model, predict_image


CHECKPOINT_PATH = Path("models/best_model.pt")
METRICS_PATH = Path("models/metrics.json")


st.set_page_config(page_title="Breast Tumor Recognition", page_icon=":material/biotech:", layout="wide")


def read_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def cached_model(path: str):
    return load_checkpoint_model(Path(path))


def render_metrics(metrics: dict | None) -> None:
    st.subheader("Model Evaluation")
    if metrics is None:
        st.info("Train a model first to show accuracy, F1-score, and confusion matrix here.")
        return

    test = metrics.get("test", {})
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{test.get('accuracy', 0.0):.3f}")
    cols[1].metric("Macro F1", f"{test.get('macro_f1', 0.0):.3f}")
    cols[2].metric("Malignant Recall", f"{test.get('per_class', {}).get('malignant', {}).get('recall', 0.0):.3f}")
    cols[3].metric("Model", metrics.get("model_name", "unknown"))

    matrix = test.get("confusion_matrix")
    if matrix:
        st.caption("Confusion matrix: rows are true labels, columns are predicted labels.")
        st.table({"class": CLASS_NAMES, **{name: [row[i] for row in matrix] for i, name in enumerate(CLASS_NAMES)}})


def main() -> None:
    st.title("Breast Tumor Ultrasound Recognition")
    st.caption("Course demo: transfer learning classifier with Grad-CAM visual explanation.")

    with st.sidebar:
        st.header("Demo Controls")
        st.write("Upload a breast ultrasound image to run inference.")
        st.divider()
        st.warning("For coursework demonstration only. Not for medical diagnosis.")

    if not CHECKPOINT_PATH.exists():
        st.error("No trained checkpoint found at models/best_model.pt.")
        st.code("python train.py --model resnet18 --epochs 12", language="bash")
        render_metrics(read_metrics())
        return

    model, meta, preprocess = cached_model(str(CHECKPOINT_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    uploaded = st.file_uploader("Upload ultrasound image", type=["png", "jpg", "jpeg", "bmp"])
    left, right = st.columns([1, 1])

    if uploaded is None:
        left.info("Waiting for an uploaded image.")
        right.empty()
    else:
        image = Image.open(uploaded).convert("RGB")
        probs, pred_idx = predict_image(model, image, preprocess, device)
        pred_name = meta["class_names"][pred_idx]

        left.subheader("Input Image")
        left.image(image, use_container_width=True)

        right.subheader("Prediction")
        right.metric("Predicted Class", pred_name, f"{float(probs[pred_idx]) * 100:.1f}% confidence")
        for idx, class_name in enumerate(meta["class_names"]):
            probability = float(probs[idx])
            right.write(f"{class_name}: {probability:.1%}")
            right.progress(probability)

        try:
            overlay = make_gradcam_overlay(model, image, preprocess, pred_idx, device, meta["model_name"])
            st.subheader("Grad-CAM Explanation")
            st.image(np.hstack([np.asarray(image.resize(overlay.size)), np.asarray(overlay)]), caption="Original image | Grad-CAM overlay")
        except Exception as exc:
            st.info(f"Grad-CAM could not be generated for this image: {exc}")

    render_metrics(read_metrics())


if __name__ == "__main__":
    main()
