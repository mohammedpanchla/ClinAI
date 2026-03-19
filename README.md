---
title: ClinAI - Multi-Model Clinical Diagnostic System
emoji: 🧠
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🧠 ClinAI

### End-to-End Multi-Model Clinical Diagnostic System

**VGG16 · U-Net · CNN+LSTM · FastAPI · Docker**

*Built by [Muhammed Panchla](https://www.linkedin.com/in/flowgenix-ai-b51517278) · Flowgenix AI*

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=flat-square&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=flat-square)

</div>

---

## Overview

**ClinAI** is a production-ready, end-to-end clinical AI platform that integrates five independent deep learning models into a single unified FastAPI backend. A clinician (or researcher) submits patient data — brain MRI, chest ECG, or risk factors — and receives a structured, PDF clinical report covering five diagnostic domains simultaneously.

The system is fully containerised with Docker and deployed as a public Space on Hugging Face. This is not a demo or Jupyter notebook — it is a deployable clinical AI backend designed to mirror real-world medical AI pipeline architecture.

---

## Clinical Problem Statement

Medical diagnosis is fragmented. A patient presenting with neurological symptoms may need brain imaging analysis, cardiac risk assessment, and stroke risk stratification — but each requires different specialists and systems. ClinAI demonstrates that a unified AI backend can serve multiple diagnostic functions from a single API, reducing integration overhead and providing structured clinical outputs in one workflow.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         ClinAI FastAPI Backend       │
                    │                                     │
   [MRI Image] ────►│  Brain Tumor Detection  (VGG16)     │
                    │  Brain Tumor Segmentation (U-Net)   │
   [ECG Signal] ───►│  ECG Classification  (CNN + LSTM)   │
                    │                                     │
   [Risk Factors] ─►│  Heart Disease Risk  (Logistic Reg) │
                    │  Stroke Risk         (Logistic Reg) │
                    │                                     │
                    │         Pipeline Orchestrator        │
                    │               ↓                     │
                    │      PDF Report Generator           │
                    │       (WeasyPrint + HTML)           │
                    └─────────────────────────────────────┘
```

### Model Components

| Module | Architecture | Task | Classes/Output |
|---|---|---|---|
| **Brain Tumor Detection** | VGG16 (fine-tuned) | 4-class classification | Glioma / Meningioma / Pituitary / No Tumor |
| **Brain Tumor Segmentation** | U-Net + VGG16 encoder | Pixel-wise segmentation | Tumor mask overlay |
| **ECG Classification** | CNN + LSTM hybrid | Heartbeat abnormality detection | 5-class arrhythmia |
| **Heart Disease Risk** | Logistic Regression + preprocessing pipeline | Binary risk prediction | Risk score + probability |
| **Stroke Risk** | Logistic Regression + preprocessing pipeline | Binary risk prediction | Risk score + probability |

### Why This Architecture?

**VGG16 for tumor detection** — VGG16's deep convolutional stack extracts rich spatial hierarchies from MRI images. Fine-tuned on brain MRI data, the feature representations transfer directly to tumor morphology.

**U-Net for segmentation** — The encoder-decoder skip connections in U-Net preserve spatial resolution lost during downsampling, enabling precise pixel-level tumor boundary delineation. VGG16 as the encoder provides pretrained feature extraction, reducing the data requirement for training segmentation from scratch.

**CNN + LSTM for ECG** — ECG signals are sequential. The CNN extracts local morphological features (QRS complex shape, P-wave), while the LSTM captures temporal dependencies across heartbeat intervals. This hybrid is the standard architecture for ECG classification in clinical AI literature.

---

## Output

ClinAI produces two output formats from a single API call:

**1. Interactive Web Results** — real-time JSON response rendered in the frontend with prediction labels, confidence scores, and segmentation overlays.

**2. Structured PDF Clinical Report** — generated via WeasyPrint from an HTML template. The report includes patient-facing findings, model predictions with confidence values, and a structured findings summary per diagnostic module.

---

## Project Structure

```
ClinAI/
├── app/
│   ├── models/                  # Individual model wrapper classes
│   │   ├── tumor_detector.py    # VGG16 brain tumor detection
│   │   ├── tumor_segmentor.py   # U-Net segmentation
│   │   ├── ecg_classifier.py    # CNN+LSTM ECG model
│   │   ├── heart_risk.py        # Heart disease logistic model
│   │   └── stroke_risk.py       # Stroke risk logistic model
│   ├── services/
│   │   ├── pipeline.py          # Orchestrates all model inference
│   │   └── report_generator.py  # HTML → PDF via WeasyPrint
│   └── templates/
│       └── report.html          # PDF report template
├── static/
│   └── index.html               # Frontend UI
├── weights/                     # Trained model artifacts (see below)
├── main.py                      # FastAPI app + static serving
├── requirements.txt
└── Dockerfile
```

---

## Model Weights

Place all model files inside `weights/` with **exact filenames**:

### PyTorch (`.pth`)
| File | Model |
|---|---|
| `tumor_detection.pth` | VGG16 brain tumor classifier |
| `brain_tumor_segmentation_best_model.pth` | U-Net segmentation |
| `ecg_classifier.pth` | CNN+LSTM ECG classifier |

### Scikit-learn / Pickle (`.pkl`)
| File | Purpose |
|---|---|
| `heart_risk_model.pkl` | Heart disease logistic regression |
| `heart_scaler.pkl` | Heart feature scaler |
| `heart_columns.pkl` | Heart feature column order |
| `stroke_risk_model.pkl` | Stroke risk logistic regression |
| `stroke_scaler.pkl` | Stroke feature scaler |
| `stroke_columns.pkl` | Stroke feature column order |

> If any filename differs, the app will fail at model loading startup.

---

## Local Setup

```bash
git clone https://github.com/muhammedpanchla/ClinAI.git
cd ClinAI
pip install -r requirements.txt
python main.py
```

Open: `http://127.0.0.1:7860`

---

## Docker

```bash
docker build -t clinai .
docker run -p 7860:7860 clinai
```

---

## Deployment Notes (Hugging Face Space)

- Configured as a **Docker Space** via `Dockerfile`
- WeasyPrint system libraries installed in Dockerfile for PDF generation
- `requirements.txt` uses **CPU-only PyTorch wheels** (required for Space CPU runtime)
- First build takes time due to PyTorch + CV dependency resolution
- All model weights must be committed to the Space repository under `weights/`

---

## Limitations

- Brain tumor detection validated on public MRI dataset — performance may degrade on scanner-specific acquisition protocols
- ECG classifier trained on standard benchmark data — not validated on raw clinical ECG recordings
- Risk models use logistic regression on tabular data — not intended as standalone clinical tools
- Not a certified medical device. All outputs are for research purposes only.

---

## ⚠️ Disclaimer

> **Research use only.** ClinAI is not a certified medical device and has not undergone clinical validation. All model outputs are for research and educational purposes. No output from this system should be used in clinical decision-making without review by a qualified medical professional. The author accepts no liability for clinical misuse.

---

## Author

**Muhammed Panchla** — Flowgenix AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Flowgenix_AI-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/flowgenix-ai-b51517278)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-muhammedpanchla-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/muhammedpanchla)
[![GitHub](https://img.shields.io/badge/GitHub-muhammedpanchla-181717?style=flat-square&logo=github)](https://github.com/muhammedpanchla)

---

<div align="center">
<sub>Built by Muhammed Panchla · Flowgenix AI · 2026</sub>
</div>
