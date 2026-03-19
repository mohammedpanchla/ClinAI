"""
Diagnostic Pipeline
───────────────────
Orchestrates all five models into one unified result dictionary.
This is the single source of truth that feeds both the web UI and the PDF report.

Flow:
    1. Run TumorDetector     on detection image (required)
    2. Run TumorSegmentor    on segmentation image (optional)
    3. Run ECGClassifier     on ECG CSV (required)
    4. Run HeartRiskPredictor on clinical numbers (required)
    5. Run StrokeRiskPredictor on clinical numbers (required)
    6. Merge all outputs → DiagnosticResult dict
    7. Compute overall_status for the executive summary
"""

import os
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from app.models import (
    TumorDetector,
    TumorSegmentor,
    ECGClassifier,
    HeartRiskPredictor,
    StrokeRiskPredictor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton model registry — models load once at startup
# ─────────────────────────────────────────────────────────────────────────────
class ModelRegistry:
    _instance = None

    def __init__(self):
        base = os.path.join(os.path.dirname(__file__), "..", "..", "weights")
        self.enable_risk_models = os.getenv("CLINAI_ENABLE_RISK_MODELS", "0") == "1"

        self.tumor_detector  = TumorDetector(
            weights_path=os.path.join(base, "tumor_detection.pth")
        )
        self.tumor_segmentor = TumorSegmentor(
            weights_path=os.path.join(base, "brain_tumor_segmentation_best_model.pth")
        )
        self.ecg_classifier  = ECGClassifier(
            weights_path=os.path.join(base, "ecg_classifier.pth")
        )
        self.heart_predictor = None
        self.stroke_predictor = None

        if not self.enable_risk_models:
            print("[compat] Risk model artifacts disabled; heuristic fallback will be used")
            return

        try:
            self.heart_predictor = HeartRiskPredictor(
                model_path  =os.path.join(base, "heart_risk_model.pkl"),
                scaler_path =os.path.join(base, "heart_scaler.pkl"),
                columns_path=os.path.join(base, "heart_columns.pkl"),
            )
        except Exception as e:
            print(f"[compat] Heart predictor disabled; fallback will be used: {e}")

        try:
            self.stroke_predictor = StrokeRiskPredictor(
                model_path  =os.path.join(base, "stroke_risk_model.pkl"),
                scaler_path =os.path.join(base, "stroke_scaler.pkl"),
                columns_path=os.path.join(base, "stroke_columns.pkl"),
            )
        except Exception as e:
            print(f"[compat] Stroke predictor disabled; fallback will be used: {e}")

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ─────────────────────────────────────────────────────────────────────────────
# Overall status logic
# ─────────────────────────────────────────────────────────────────────────────
def _overall_status(tumor_result: dict, ecg_result: dict,
                    heart_result: dict, stroke_result: dict) -> dict:
    """
    Derive a single system-wide alert level from all model outputs.
    Rules (highest wins):
        CRITICAL  → HIGH severity tumor OR two+ HIGH risk signals
        WARNING   → Any abnormal ECG OR MODERATE/HIGH risk in one system
        NORMAL    → Everything clear
    """
    flags = []

    if tumor_result["severity"] == "HIGH":
        flags.append("CRITICAL")
    elif tumor_result["severity"] == "MODERATE":
        flags.append("WARNING")

    if ecg_result["label"] == "ABNORMAL":
        flags.append("WARNING")

    if heart_result["risk_level"] == "HIGH":
        flags.append("CRITICAL" if "CRITICAL" in flags else "WARNING")
    elif heart_result["risk_level"] == "MODERATE":
        flags.append("WARNING")

    if stroke_result["risk_level"] == "HIGH":
        flags.append("CRITICAL" if "CRITICAL" in flags else "WARNING")
    elif stroke_result["risk_level"] == "MODERATE":
        flags.append("WARNING")

    if "CRITICAL" in flags:
        return {"status": "CRITICAL", "color": "#e74c3c",
                "message": "Critical findings detected. Immediate specialist consultation recommended."}
    elif "WARNING" in flags:
        return {"status": "WARNING",  "color": "#e67e22",
                "message": "Abnormal findings detected. Medical review advised."}
    else:
        return {"status": "NORMAL",   "color": "#2ecc71",
                "message": "No critical findings detected. Routine follow-up recommended."}


def _risk_band(prob: float) -> dict:
    if prob < 0.30:
        return {"level": "LOW",      "color": "#00c97d", "label": "Low Risk",      "risk_label": "Low Risk"}
    elif prob < 0.60:
        return {"level": "MODERATE", "color": "#f59e0b", "label": "Moderate Risk", "risk_label": "Moderate Risk"}
    else:
        return {"level": "HIGH",     "color": "#e8192c", "label": "High Risk",     "risk_label": "High Risk"}


def _fallback_heart_risk(
    patient_age: int,
    resting_bp: float,
    cholesterol: float,
    fasting_bs: int,
    max_hr: float,
    oldpeak: float,
    exercise_angina: int,
) -> dict:
    """
    Fallback heuristic if sklearn artifact compatibility breaks at runtime.
    """
    score = 0.05
    score += 0.20 if patient_age >= 55 else 0.10 if patient_age >= 45 else 0.0
    score += 0.15 if resting_bp >= 140 else 0.07 if resting_bp >= 130 else 0.0
    score += 0.10 if cholesterol >= 240 else 0.05 if cholesterol >= 200 else 0.0
    score += 0.10 if int(fasting_bs) == 1 else 0.0
    score += 0.15 if max_hr < 120 else 0.07 if max_hr < 140 else 0.0
    score += 0.10 if oldpeak >= 1.5 else 0.05 if oldpeak >= 1.0 else 0.0
    score += 0.15 if int(exercise_angina) == 1 else 0.0
    prob = max(0.01, min(0.95, score))
    band = _risk_band(prob)
    return {"probability": round(prob, 4), "risk_level": band["level"], **band, "fallback": True}


def _fallback_stroke_risk(
    patient_age: int,
    avg_glucose_level: float,
    hypertension: int,
    heart_disease_history: int,
    smoking_status: str,
) -> dict:
    """
    Fallback heuristic if sklearn artifact compatibility breaks at runtime.
    """
    score = 0.01
    score += 0.15 if patient_age > 60 else 0.10 if patient_age >= 45 else 0.02
    score += 0.20 if int(hypertension) == 1 else 0.0
    score += 0.15 if int(heart_disease_history) == 1 else 0.0
    score += 0.12 if avg_glucose_level >= 160 else 0.06 if avg_glucose_level >= 126 else 0.0
    score += 0.10 if smoking_status in ("smokes", "formerly smoked") else 0.0
    prob = max(0.005, min(0.90, score))
    band = _risk_band(prob)
    return {"probability": round(prob, 4), "risk_level": band["level"], **band, "fallback": True}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostic_pipeline(
    # Patient info
    patient_name: str,
    patient_age: int,
    patient_gender: str,

    # Files
    detection_image_bytes: bytes,
    detection_image_ext: str,
    ecg_csv_bytes: bytes,

    # Clinical numbers for risk models
    resting_bp: float,
    cholesterol: float,
    max_hr: float,
    fasting_bs: int,
    oldpeak: float,
    chest_pain_type: str,         # "ATA" | "NAP" | "TA" | "ASY"
    resting_ecg: str,             # "Normal" | "ST" | "LVH"
    exercise_angina: int,         # 0 | 1
    st_slope: str,                # "Up" | "Flat" | "Down"

    bmi: float,
    avg_glucose_level: float,
    hypertension: int,            # 0 | 1
    heart_disease_history: int,   # 0 | 1
    ever_married: int,            # 0 | 1
    is_urban: int,                # 0 | 1
    work_type: str,               # "Private"|"Self-employed"|"Govt_job"|"children"|"Never_worked"
    smoking_status: str,          # "formerly smoked"|"never smoked"|"smokes"|"Unknown"

    # Optional segmentation
    segmentation_image_bytes: Optional[bytes] = None,
    segmentation_image_ext: str = "tif",

) -> dict:
    """
    Run all models and return a single unified result dictionary.
    """
    registry = ModelRegistry.get()
    report_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%d %B %Y, %H:%M")

    # ── 1. Tumor Detection ────────────────────────────────────────────────────
    tumor_result = registry.tumor_detector.predict(detection_image_bytes)

    # ── 2. Tumor Segmentation (optional) ─────────────────────────────────────
    seg_result = None
    if segmentation_image_bytes:
        seg_result = registry.tumor_segmentor.predict(
            segmentation_image_bytes, segmentation_image_ext
        )

    # ── 3. ECG Classification ─────────────────────────────────────────────────
    ecg_result = registry.ecg_classifier.predict(ecg_csv_bytes)

    # ── 4. Heart Risk ─────────────────────────────────────────────────────────
    sex_raw = "M" if patient_gender.upper() in ("M", "MALE") else "F"
    angina_raw = "Y" if int(exercise_angina) == 1 else "N"

    heart_input = {
        "Age":                  patient_age,
        "Sex":                  sex_raw,
        "RestingBP":            resting_bp,
        "Cholesterol":          cholesterol,
        "FastingBS":            fasting_bs,
        "MaxHR":                max_hr,
        "ExerciseAngina":       angina_raw,
        "Oldpeak":              oldpeak,
        "ChestPainType":        chest_pain_type,
        "RestingECG":           resting_ecg,
        "ST_Slope":             st_slope,
    }
    if registry.heart_predictor is not None:
        try:
            heart_result = registry.heart_predictor.predict(heart_input)
        except Exception as e:
            print(f"[compat] Heart predictor runtime failure; using fallback: {e}")
            heart_result = _fallback_heart_risk(
                patient_age=patient_age,
                resting_bp=resting_bp,
                cholesterol=cholesterol,
                fasting_bs=fasting_bs,
                max_hr=max_hr,
                oldpeak=oldpeak,
                exercise_angina=exercise_angina,
            )
            heart_result["model_error"] = str(e)
    else:
        heart_result = _fallback_heart_risk(
            patient_age=patient_age,
            resting_bp=resting_bp,
            cholesterol=cholesterol,
            fasting_bs=fasting_bs,
            max_hr=max_hr,
            oldpeak=oldpeak,
            exercise_angina=exercise_angina,
        )
        heart_result["model_error"] = "Heart predictor unavailable; fallback used"

    # ── 5. Stroke Risk (v2 — raw values, new ColumnTransformer model) ───────────
    # Convert int flags back to the string format the new OHE pipeline expects
    gender_str    = "Male" if patient_gender.upper() in ("M", "MALE") else "Female"
    married_str   = "Yes" if ever_married == 1 else "No"
    residence_str = "Urban" if is_urban == 1 else "Rural"

    stroke_input = {
        "age":              patient_age,
        "avg_glucose_level": avg_glucose_level,
        # bmi intentionally omitted — v3 model was trained without it (DROP_BMI=True)
        "gender":           gender_str,
        "hypertension":     hypertension,       # 0 | 1  (int)
        "heart_disease":    heart_disease_history,  # 0 | 1  (int)
        "ever_married":     married_str,        # "Yes" | "No"
        "work_type":        work_type,          # raw string
        "Residence_type":   residence_str,      # "Urban" | "Rural"
        "smoking_status":   smoking_status,     # raw string
    }
    if registry.stroke_predictor is not None:
        try:
            stroke_result = registry.stroke_predictor.predict(stroke_input)
        except Exception as e:
            print(f"[compat] Stroke predictor runtime failure; using fallback: {e}")
            stroke_result = _fallback_stroke_risk(
                patient_age=patient_age,
                avg_glucose_level=avg_glucose_level,
                hypertension=hypertension,
                heart_disease_history=heart_disease_history,
                smoking_status=smoking_status,
            )
            stroke_result["model_error"] = str(e)
    else:
        stroke_result = _fallback_stroke_risk(
            patient_age=patient_age,
            avg_glucose_level=avg_glucose_level,
            hypertension=hypertension,
            heart_disease_history=heart_disease_history,
            smoking_status=smoking_status,
        )
        stroke_result["model_error"] = "Stroke predictor unavailable; fallback used"

    # ── 6. Overall Status ─────────────────────────────────────────────────────
    overall = _overall_status(tumor_result, ecg_result, heart_result, stroke_result)

    # ── 7. Assemble unified result ────────────────────────────────────────────
    return {
        "report_id":   report_id,
        "timestamp":   timestamp,
        "patient": {
            "name":    patient_name,
            "age":     patient_age,
            "gender":  patient_gender,
        },
        "overall":     overall,
        "tumor":       tumor_result,
        "segmentation": seg_result,
        "ecg":         ecg_result,
        "heart":       heart_result,
        "stroke":      stroke_result,
        "clinical": {
            "resting_bp":       resting_bp,
            "cholesterol":      cholesterol,
            "max_hr":           max_hr,
            "fasting_bs":       fasting_bs,
            "bmi":              bmi,
            "avg_glucose":      avg_glucose_level,
            "hypertension":     hypertension,
            "smoking_status":   smoking_status,
        }
    }
