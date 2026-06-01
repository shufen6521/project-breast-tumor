# Breast Tumor Recognition Presentation Demo

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

## 5. Run the FastAPI Backend

The backend reuses the trained PyTorch checkpoint and exposes prediction APIs
for a React frontend.

```bash
uvicorn backend.main:app --reload
```

API endpoints:

- `GET /health`
- `GET /metrics`
- `POST /predict` with an uploaded image file named `file`

Open the interactive API docs at:

```text
http://127.0.0.1:8000/docs
```

For GitHub, keep `models/best_model.pt` out of normal commits. Upload the
checkpoint to a GitHub Release and document that users should place it at:

```text
models/best_model.pt
```

## 6. Run the React Frontend

The React frontend is a medical imaging workstation for uploading ultrasound
images, calling the FastAPI backend, and showing prediction probabilities plus
Grad-CAM explanations.

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

If the backend runs on another address, create `frontend/.env.local`:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Presentation Notes

This project is for AI/ML coursework and demonstration only. It is not a
medical diagnostic system.
