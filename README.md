# 🐱🐶 Cats vs. Dogs Image Classifier

[![CI/CD](https://github.com/yesahek/CatsVsDogs-Image-Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/yesahek/CatsVsDogs-Image-Classifier/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)


> **Live Demo →** [cats-vs-dogs.onrender.com](https://cats-vs-dogs.onrender.com)

A production-grade deep learning pipeline that classifies images of cats and dogs.
Built from a custom CNN baseline through VGG16 transfer learning, deployed as a Flask web app with automated CI/CD.

---

## Results

| Model | Val Accuracy | Test Accuracy | AUC | Params |
|---|---|---|---|---|
| Custom CNN (scratch) | ~80% | ~78% | ~0.870 | ~2M |
| **VGG16 (fine-tuned) ⭐** | **~98%** | **~97%** | **~0.998** | **138M** |
| ResNet50 | ~96% | ~95% | ~0.990 | 25M |
| EfficientNetB0 | ~95% | ~94% | ~0.988 | 5.3M |

⭐ Production model

---

## Demo

**Single image prediction**

```bash
curl -X POST -F "image=@cat.jpg" https://cats-vs-dogs.onrender.com/predict
```

```json
{
  "class": "Cat",
  "confidence": 0.9832,
  "percent": "98.3%",
  "emoji": "🐱",
  "prob_dog": 0.0168
}
```

**Batch prediction**

```bash
curl -X POST -F "zipfile=@images.zip" https://cats-vs-dogs.onrender.com/predict/batch
```

**Health check**

```bash
curl https://cats-vs-dogs.onrender.com/health
# {"status": "ok", "model": "models/vgg16_phase2_best.h5"}
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yesahek/CatsVsDogs-Image-Classifier.git
cd CatsVsDogs-Image-Classifier

# 2. Environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download dataset (requires Kaggle API key in config/kaggle.json)
python src/data_loader.py

# 4. Train
python train.py --model vgg16 --epochs 20

# 5. Evaluate
python eval.py --model models/vgg16_phase2_best.h5

# 6. Run web app
python app/app.py
# → http://localhost:5000
```

---

## Project Structure

```
CatsVsDogs-Image-Classifier/
│
├── notebooks/                       # Step-by-step Jupyter notebooks
│   ├── data_loader.ipynb            # Kaggle download + Drive setup
│   ├── eda.ipynb                    # Exploratory data analysis
│   ├── split_dataset.ipynb          # Train / val / test split
│   ├── preprocess.ipynb             # Resize, normalise, augment, tf.data
│   ├── models.ipynb                 # 4 model architectures
│   ├── training.ipynb               # Train + save weights
│   └── evaluation.ipynb             # Test set metrics + plots
│
├── src/                             # Core Python modules
│   ├── data_loader.py               # Dataset download, split, tf.data pipeline
│   ├── models.py                    # CNN + transfer learning architectures
│   ├── train_utils.py               # Callbacks, compile, training loop
│   └── evaluate.py                  # Metrics, confusion matrix, plots
│
├── app/                             # Flask web application
│   ├── app.py                       # 4 endpoints: /, /predict, /predict/batch, /health
│   ├── templates/index.html         # Upload UI (3 tabs)
│   └── static/
│       ├── css/style.css
│       └── js/app.js
│
├── tests/                           # Automated tests
│   ├── test_preprocessing.py        # Unit tests (preprocessing + output format)
│   ├── test_api.py                  # Integration tests (all Flask endpoints)
│   └── test_smoke.py                # Post-deploy smoke tests (live URL)
│
├── docs/                            # Evaluation reports + plots
│   ├── evaluation-and-design.md
│   └── ai-tooling.md
│
├── .github/workflows/
│   └── ci.yml                       # GitHub Actions: test → deploy → smoke test
│
├── train.py                         # Training entry point
├── eval.py                          # Evaluation entry point
├── requirements.txt
└── README.md
```

---

## Architecture

### Data Pipeline

```
data/raw/
├── Cat/  (12,500 images)
└── Dog/  (12,500 images)
        │
        ▼  split_dataset()
data/processed/
├── train/  (~16,000)   ← augmentation applied here only
├── val/    (~4,000)    ← clean images
└── test/   (~5,000)    ← never seen during training
        │
        ▼  tf.data pipeline
resize 224×224 → normalise [0,1] → augment (train only) → cache → prefetch
```

**Augmentation (training only):** horizontal flip · ±20° rotation · ±15% zoom · ±10% translation

### Transfer Learning Strategy

```
Phase 1 — Frozen base  (5 epochs,  LR = 1e-3)
  Pretrained base : LOCKED
  Custom head     : TRAINS  (GlobalAvgPool → Dense(256) → Dropout → Dense(1, sigmoid))

Phase 2 — Fine-tuning  (15 epochs, LR = 1e-5)
  Top 20 base layers : UNLOCKED
  Custom head        : TRAINS
```

### Web App Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Upload UI |
| POST | `/predict` | Single image → class + confidence |
| POST | `/predict/batch` | ZIP of images → predictions table |
| GET | `/health` | `{"status": "ok"}` for CI/CD |

---

## CI/CD Pipeline

Every push to `main` triggers three automated jobs:

```
Push to main
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  1. Test    │────▶│  2. Deploy   │────▶│  3. Smoke   │
│             │     │              │     │  Test       │
│ unit tests  │     │ Render hook  │     │ GET /health │
│ integration │     │              │     │ POST /predict│
│ tests       │     │              │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
```

Deploy and smoke test only run if all tests pass.

---

## Running Tests

```bash
# Unit tests
pytest tests/test_preprocessing.py -v

# Integration tests (no model file needed — model is mocked)
pytest tests/test_api.py -v

# All tests
pytest tests/ -v --ignore=tests/test_smoke.py

# Smoke tests against live URL
APP_URL=https://cats-vs-dogs.onrender.com pytest tests/test_smoke.py -v
```

---

## Deployment

The app is deployed on [Render](https://render.com) (free tier).

**Environment variables required:**

| Variable | Description |
|---|---|
| `MODEL_PATH` | Path to saved `.h5` model |
| `PORT` | Server port (Render sets this automatically) |

**GitHub Secrets required for CI/CD:**

| Secret | Description |
|---|---|
| `RENDER_DEPLOY_HOOK` | Render deploy hook URL |
| `APP_URL` | Live app URL for smoke tests |

---

## Tech Stack

| Category | Technology |
|---|---|
| Deep learning | TensorFlow 2.x · Keras |
| Models | VGG16 · ResNet50 · EfficientNetB0 · Custom CNN |
| Web framework | Flask 3.0 |
| Data | NumPy · Pillow · scikit-learn |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Deployment | Render |
| Dataset | [Microsoft Cats and Dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) |

---

## Notebooks

All notebooks run on Google Colab with a T4 GPU (free):

| Notebook | Open in Colab |
|---|---|
| Data Loader | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yesahek/CatsVsDogs-Image-Classifier/blob/main/notebooks/data_loader.ipynb) |
| EDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yesahek/CatsVsDogs-Image-Classifier/blob/main/notebooks/eda.ipynb) |
| Split Dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yesahek/CatsVsDogs-Image-Classifier/blob/main/notebooks/split_dataset.ipynb) |
| Preprocess | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yesahek/CatsVsDogs-Image-Classifier/blob/main/notebooks/preprocess.ipynb) |
| Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yesahek/CatsVsDogs-Image-Classifier/blob/main/notebooks/models.ipynb) |
| Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yesahek/CatsVsDogs-Image-Classifier/blob/main/notebooks/training.ipynb) |
| Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yesahek/CatsVsDogs-Image-Classifier/blob/main/notebooks/evaluation.ipynb) |

---

