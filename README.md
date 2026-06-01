# Breast Tumor Recognition 

This project is a course-ready breast ultrasound image recognition demo. It uses
PyTorch transfer learning for a three-class BUSI task:

- `normal`
- `benign`
- `malignant`

The frontend is a Streamlit app, so no separate backend is required.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── train.py
├── data/
│   ├── raw/BUSI/
│   └── processed/
├── models/
└── src/
    ├── config.py
    ├── data.py
    ├── gradcam.py
    ├── inference.py
    ├── metrics.py
    └── models.py
```

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If PyTorch is difficult to install on your machine, use the official selector:
https://pytorch.org/get-started/locally/

## 2. Prepare BUSI Dataset

Download the BUSI dataset and place images like this:

```text
data/raw/BUSI/
├── benign/
├── malignant/
└── normal/
```

Files containing `mask` in their name are automatically ignored for
classification training.

Recommended dataset references for the presentation:

- BUSI paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/
- Hugging Face mirror: https://huggingface.co/datasets/Angelou0516/BUSI

## 3. Train

Train a ResNet18 baseline:

```bash
python train.py --model resnet18 --epochs 12
```

Train EfficientNet-B0:

```bash
python train.py --model efficientnet_b0 --epochs 12
```

If pretrained weights cannot be downloaded because the machine has no internet,
use:

```bash
python train.py --model resnet18 --epochs 12 --no-pretrained
```

The best checkpoint is saved to:

```text
models/best_model.pt
```

Metrics are saved to:

```text
models/metrics.json
```

## 4. Run the Demo

```bash
streamlit run app.py
```

Upload a breast ultrasound image. The app shows:

- predicted class
- class probabilities
- Grad-CAM heatmap
- test-set metrics from the latest training run

## Presentation Notes

This project is for AI/ML coursework and demonstration only. It is not a
medical diagnostic system.
