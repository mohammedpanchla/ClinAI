<div align="center">

# 🧠 ClinAI
### Multi-Model Clinical Diagnostic System

**VGG16 · U-Net · CNN+LSTM · Logistic Regression · FastAPI · Docker**

*Built by [Muhammed Panchla](https://www.linkedin.com/in/flowgenix-ai-b51517278) · Flowgenix AI*

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/muhammedpanchla/ClinAI)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=flat-square)]()

---

> **⚠️ Research use only.** ClinAI is not a certified medical device. All outputs are for research and educational purposes only. No output should be used in clinical decision-making without review by a qualified medical professional.

</div>

---

## 📌 What is ClinAI?

ClinAI is a production-ready, end-to-end clinical AI platform that integrates **five independent deep learning and machine learning models** into a single unified FastAPI backend.

A user submits patient data — a brain MRI scan, an ECG signal CSV, and clinical risk factors — and receives:

- A **real-time JSON response** with all model predictions
- A **structured PDF clinical report** covering five diagnostic domains simultaneously

This is not a Jupyter notebook demo. It is a deployable AI backend designed to mirror real-world medical AI pipeline architecture — with clean separation of concerns, a model registry singleton, graceful fallbacks, and a report generation service.

---

## 🔴 Live Demo

| Platform | Link |
|---|---|
| 🤗 Hugging Face Space (Live App) | [huggingface.co/spaces/muhammedpanchla/ClinAI](https://huggingface.co/spaces/muhammedpanchla/ClinAI) |
| 📄 Example Generated PDF Report | [pdflink.to/clinai](https://pdflink.to/clinai/) |
| 🧪 Model Weights Repository | [huggingface.co/muhammedpanchla/ClinAI](https://huggingface.co/muhammedpanchla/ClinAI/tree/main) |

---

## 🏗️ System Architecture

```
                         Patient Data Input
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
         [MRI Image]     [ECG CSV File]    [Clinical Vitals]
              │                │                 │
              └────────────────┼─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   FastAPI Backend   │
                    │      main.py        │
                    │  POST /api/analyze  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    pipeline.py      │
                    │  ModelRegistry      │
                    │  (Singleton)        │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         │           │         │          │            │
    ┌────▼────┐ ┌────▼────┐ ┌─▼───────┐ ┌▼──────┐ ┌──▼─────┐
    │ VGG16   │ │ U-Net   │ │CNN+LSTM │ │LogReg │ │LogReg  │
    │Detection│ │Segment. │ │  ECG    │ │ Heart │ │ Stroke │
    └────┬────┘ └────┬────┘ └─┬───────┘ └┬──────┘ └──┬─────┘
         │           │         │          │            │
         └─────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Unified Result Dict │
                    │  + Overall Status   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼──────────────────┐
              │                                   │
   ┌──────────▼──────────┐           ┌────────────▼────────┐
   │   JSON Response     │           │  report_generator   │
   │  (all predictions)  │           │  Jinja2 + WeasyPrint│
   └─────────────────────┘           │  → PDF Clinical     │
                                     │    Report           │
                                     └─────────────────────┘
```

---

## 🤖 Models

| Model | Architecture | Task | Output |
|---|---|---|---|
| Brain Tumor Detection | VGG16 (fine-tuned) | 4-class classification | Glioma / Meningioma / Pituitary / No Tumor + confidence |
| Brain Tumor Segmentation | U-Net + VGG16 encoder | Pixel-wise segmentation | Binary mask + overlay + heatmap |
| ECG Classification | CNN + LSTM hybrid | Heartbeat abnormality | Normal / Abnormal + waveform chart |
| Heart Disease Risk | Logistic Regression | Binary risk prediction | Risk probability + risk level |
| Stroke Risk | Logistic Regression | Binary risk prediction | Risk probability + risk level |

### Architecture Decisions

**VGG16 for tumor detection** — VGG16's deep convolutional stack (13 layers) extracts rich spatial hierarchies from MRI images. Pretrained on ImageNet and fine-tuned on brain MRI data, the feature representations transfer directly to tumor morphology recognition. Achieved **97.14% validation accuracy** on a 4-class balanced dataset.

**U-Net for segmentation** — Encoder-decoder skip connections preserve spatial resolution lost during downsampling, enabling precise pixel-level tumor boundary delineation. VGG16 used as the encoder provides pretrained feature extraction. Differential learning rates applied: encoder at `1e-5` (preserve pretrained weights), decoder at `5e-4` (train from scratch). **Test Dice: 81.2%, IoU: 73.9%**.

**CNN + LSTM for ECG** — ECG signals are sequential time series. The 1D CNN extracts local morphological features (QRS complex shape, P-wave, T-wave), while the LSTM captures temporal dependencies across heartbeat intervals. This hybrid is the standard architecture for ECG classification in clinical AI literature. Input: 187-point signal per heartbeat.

**Logistic Regression for risk models** — Clean, interpretable, and well-suited to tabular clinical data. Each model uses a full sklearn `ColumnTransformer` pipeline (scaler + one-hot encoder) trained end-to-end, with graceful heuristic fallbacks built into the pipeline if the `.pkl` artifact fails to load.

---

## 📁 Project Structure

```
ClinAI/
│
├── app/
│   ├── models/                        # AI model wrapper classes
│   │   ├── __init__.py                # Exports all model classes
│   │   ├── tumor_detector.py          # VGG16 brain tumor detection
│   │   ├── tumor_segmentor.py         # U-Net + VGG16 segmentation
│   │   ├── ecg_classifier.py          # CNN + LSTM ECG classifier
│   │   ├── risk_predictors.py         # Heart + Stroke logistic models
│   │
│   ├── services/
│   │   ├── pipeline.py                # Orchestrates all 5 models (ModelRegistry)
│   │   └── report_generator.py        # Jinja2 HTML → WeasyPrint PDF
│   │
│   └── templates/
│       └── report.html                # Clinical PDF report template
│
├── static/
│   ├── index.html                     # Frontend UI (single page)
│   └── report-assets/                 # Icons embedded in PDF report
│       ├── brain.png
│       ├── heart.png
│       ├── chart.png
│       ├── clipboard.png
│       ├── map.png
│       └── pill.png
│
├── weights/                           # ⚠️ NOT included in this repo
│   │                                  # Download from Hugging Face — see below
│   ├── tumor_detection.pth
│   ├── brain_tumor_segmentation_best_model.pth
│   ├── ecg_classifier.pth
│   ├── heart_risk_model.pkl
│   ├── heart_scaler.pkl
│   ├── heart_columns.pkl
│   ├── stroke_risk_model.pkl
│   ├── stroke_scaler.pkl
│   └── stroke_columns.pkl
│
├── reports/                           # Auto-created at runtime (PDF output)
│
├── main.py                            # FastAPI app entry point
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker build config
└── README.md
```

---

## ⚙️ Model Weights — Download from Hugging Face

> **The `weights/` folder is not included in this GitHub repository** because model files are too large for GitHub hosting (`.pth` files range from 500MB to 1GB+).

All model weights are hosted on Hugging Face:

### 🤗 [huggingface.co/muhammedpanchla/ClinAI/tree/main](https://huggingface.co/muhammedpanchla/ClinAI/tree/main)

Download each file and place it inside the `weights/` folder with the **exact filenames listed below**. Any filename mismatch will cause the app to fail at startup.

### PyTorch weights (`.pth`)

| Filename | Model | Size |
|---|---|---|
| `tumor_detection.pth` | VGG16 brain tumor classifier | ~550 MB |
| `brain_tumor_segmentation_best_model.pth` | U-Net segmentation model | ~120 MB |
| `ecg_classifier.pth` | CNN+LSTM ECG classifier | ~15 MB |

### Scikit-learn artifacts (`.pkl`)

| Filename | Purpose |
|---|---|
| `heart_risk_model.pkl` | Heart disease logistic regression model |
| `heart_scaler.pkl` | StandardScaler for heart features |
| `heart_columns.pkl` | Feature column order for heart model |
| `stroke_risk_model.pkl` | Stroke risk logistic regression model |
| `stroke_scaler.pkl` | StandardScaler for stroke features |
| `stroke_columns.pkl` | Feature column order for stroke model |

### Quick download via `huggingface_hub`

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='muhammedpanchla/ClinAI',
    repo_type='model',
    local_dir='./weights'
)
"
```

Or download individual files:

```bash
from huggingface_hub import hf_hub_download

files = [
    'tumor_detection.pth',
    'brain_tumor_segmentation_best_model.pth',
    'ecg_classifier.pth',
    'heart_risk_model.pkl', 'heart_scaler.pkl', 'heart_columns.pkl',
    'stroke_risk_model.pkl', 'stroke_scaler.pkl', 'stroke_columns.pkl',
]

for f in files:
    hf_hub_download(
        repo_id='muhammedpanchla/ClinAI',
        repo_type='model',
        filename=f,
        local_dir='./weights'
    )
```

---

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/muhammedpanchla/ClinAI.git
cd ClinAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` uses CPU-only PyTorch wheels by default. If you have a CUDA GPU, replace the torch install line with the appropriate CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Download model weights

Follow the [Hugging Face download instructions above](#model-weights--download-from-hugging-face) and place all files in `weights/`.

### 4. Run the application

```bash
python main.py
```

Open in your browser: `http://127.0.0.1:7860`

---

## 🐳 Docker

### Build and run

```bash
docker build -t clinai .
docker run -p 7860:7860 -v $(pwd)/weights:/app/weights clinai
```

> The `-v` flag mounts your local `weights/` folder into the container so Docker doesn't need to re-download model files on every build.

### Windows (PowerShell)

```powershell
docker run -p 7860:7860 -v ${PWD}/weights:/app/weights clinai
```

---

## 📡 API Reference

### Health check

```
GET /api/health
```

Response:
```json
{
  "status": "ok",
  "system": "ClinAI v1.0"
}
```

---

### Main diagnostic endpoint

```
POST /api/analyze
Content-Type: multipart/form-data
```

#### Required fields

| Field | Type | Description |
|---|---|---|
| `patient_name` | `str` | Patient full name |
| `patient_age` | `int` | Patient age in years |
| `patient_gender` | `str` | `"M"` or `"F"` |
| `detection_image` | `File` | Brain MRI scan (JPG or PNG) |
| `ecg_csv` | `File` | ECG signal CSV — 188 rows × 1 column |
| `resting_bp` | `float` | Resting blood pressure (mmHg) |
| `cholesterol` | `float` | Serum cholesterol (mg/dL) |
| `max_hr` | `float` | Maximum heart rate achieved |
| `fasting_bs` | `int` | Fasting blood sugar > 120 mg/dL (`0` or `1`) |
| `oldpeak` | `float` | ST depression induced by exercise |
| `chest_pain_type` | `str` | `"ATA"` / `"NAP"` / `"TA"` / `"ASY"` |
| `resting_ecg` | `str` | `"Normal"` / `"ST"` / `"LVH"` |
| `exercise_angina` | `int` | `0` or `1` |
| `st_slope` | `str` | `"Up"` / `"Flat"` / `"Down"` |
| `bmi` | `float` | Body mass index |
| `avg_glucose_level` | `float` | Average glucose level (mg/dL) |
| `hypertension` | `int` | `0` or `1` |
| `heart_disease_history` | `int` | `0` or `1` |
| `ever_married` | `int` | `0` or `1` |
| `is_urban` | `int` | `0` = Rural, `1` = Urban |
| `work_type` | `str` | `"Private"` / `"Self-employed"` / `"Govt_job"` / `"children"` / `"Never_worked"` |
| `smoking_status` | `str` | `"formerly smoked"` / `"never smoked"` / `"smokes"` / `"Unknown"` |

#### Optional fields

| Field | Type | Description |
|---|---|---|
| `segmentation_image` | `File` | MRI for segmentation (TIF or PNG) — if omitted, segmentation section is skipped |

#### Response

```json
{
  "success": true,
  "report_id": "A3F92B1C",
  "pdf_url": "/reports/ClinAI_Report_A3F92B1C.pdf",
  "result": {
    "patient": { "name": "...", "age": 45, "gender": "M" },
    "overall": { "status": "WARNING", "color": "#e67e22", "message": "..." },
    "tumor":   { "tumor_class": "glioma", "confidence": 0.94, "severity": "HIGH", "all_probs": {...} },
    "segmentation": { "tumor_detected": true, "coverage_percent": 12.3, "overlay_b64": "..." },
    "ecg":     { "label": "NORMAL", "confidence": 0.87, "waveform_b64": "..." },
    "heart":   { "probability": 0.41, "risk_level": "MODERATE" },
    "stroke":  { "probability": 0.18, "risk_level": "LOW" },
    "clinical": { "resting_bp": 130, "cholesterol": 220, "..." : "..." }
  }
}
```

---

## 📊 Model Performance

| Model | Metric | Value | Dataset |
|---|---|---|---|
| Brain Tumor Detection (VGG16) | Validation Accuracy | **97.14%** | Kaggle Brain MRI (4-class) |
| Brain Tumor Detection (VGG16) | Training Accuracy | 99.35% | — |
| Brain Tumor Segmentation (U-Net) | Dice Score | **81.2%** | LGG Kaggle (kaggle_3m) |
| Brain Tumor Segmentation (U-Net) | IoU Score | **73.9%** | — |
| ECG Classification (CNN+LSTM) | Validation Accuracy | **~98%** | PTB Diagnostic ECG Database |

---

## ⚠️ Known Limitations

- Brain tumor models trained on public Kaggle MRI datasets — performance may vary across different scanner protocols and acquisition settings
- ECG classifier trained on benchmark PTB data — not validated on raw clinical ECG recordings from hospital equipment  
- Heart and stroke risk models use logistic regression on tabular data — not intended as standalone clinical risk tools
- WeasyPrint PDF generation requires system libraries (Cairo, Pango) — these are installed in the Dockerfile but require manual installation for local non-Docker setups on Linux/macOS
- Inference is CPU-based by default — expect 3–6 seconds per full pipeline on CPU; GPU dramatically reduces imaging model latency

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + Uvicorn |
| Deep learning | PyTorch 2.0+ |
| Computer vision | torchvision, segmentation-models-pytorch, albumentations, OpenCV |
| Machine learning | scikit-learn |
| Image processing | Pillow, tifffile |
| PDF generation | WeasyPrint + Jinja2 |
| Containerisation | Docker |
| Deployment | Hugging Face Spaces (Docker runtime) |
| Frontend | Vanilla HTML/CSS/JS (single file) |

---

## 🗂️ Deployment — Hugging Face Spaces

This repo is also deployed as a Hugging Face Docker Space:

- Configured via the YAML header in this README (`sdk: docker`, `app_port: 7860`)
- WeasyPrint system libraries (Cairo, Pango, GDK-Pixbuf) installed in the Dockerfile
- CPU-only PyTorch wheels used to stay within Space resource limits
- All model weights committed to the Hugging Face model repository at `muhammedpanchla/ClinAI`

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Muhammed Panchla** — Flowgenix AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Flowgenix_AI-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/flowgenix-ai-b51517278)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-muhammedpanchla-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/muhammedpanchla)
[![GitHub](https://img.shields.io/badge/GitHub-muhammedpanchla-181717?style=flat-square&logo=github)](https://github.com/muhammedpanchla)

---

<div align="center">
<sub>Built by Muhammed Panchla · Flowgenix AI · 2026</sub>
</div>
