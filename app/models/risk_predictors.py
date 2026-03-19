"""
Heart Disease & Stroke Risk Wrappers

Heart Disease  (v2 — RandomForest + ColumnTransformer)
───────────────────────────────────────────────────────
Scaler handles : Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak  (6 cols)
OHE handles    : Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope (5 cols → 14 OHE)
Total features : 20  |  Model: RandomForestClassifier  |  Threshold: 0.5

Stroke Risk  (v3 — LogisticRegression + ADASYN, recall-first, engineered features)
────────────────────────────────────────────────────────────────────────────────────
Scaler handles : age, hypertension, heart_disease, avg_glucose_level,
                 age_glucose_interact                                     (5 cols)
OHE handles    : gender, ever_married, work_type, Residence_type,
                 smoking_status, age_group, glucose_risk_bin             (7 cols → 21 OHE)
Total features : 26  |  Threshold: 0.0202  (90% recall target)
BMI            : DROPPED — model was trained without it (DROP_BMI=True)
"""

import pickle
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _risk_band(prob: float) -> dict:
    if prob < 0.30:
        return {"level": "LOW",      "color": "#00c97d", "label": "Low Risk",      "risk_label": "Low Risk"}
    elif prob < 0.60:
        return {"level": "MODERATE", "color": "#f59e0b", "label": "Moderate Risk", "risk_label": "Moderate Risk"}
    else:
        return {"level": "HIGH",     "color": "#e8192c", "label": "High Risk",     "risk_label": "High Risk"}


class _NumpyCompatUnpickler(pickle.Unpickler):
    """
    Compatibility bridge for artifacts saved with NumPy 2.x internals
    (e.g., module path `numpy._core.*`) but loaded in NumPy 1.x runtime
    (module path `numpy.core.*`).
    """
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pkl(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if "numpy._core" not in str(e):
            raise

    with open(path, "rb") as f:
        obj = _NumpyCompatUnpickler(f).load()
    print(f"[compat] Loaded {Path(path).name} via numpy._core -> numpy.core remap")
    return obj


def _safe_scale(scaler, x: np.ndarray) -> np.ndarray:
    """
    Try scaler.transform first; if sklearn version mismatch breaks transform,
    fall back to manual StandardScaler formula using persisted mean_/scale_.
    """
    orig_exc = None
    try:
        return scaler.transform(x)
    except Exception as e:
        orig_exc = e

    mean = getattr(scaler, "mean_", None)
    if mean is None:
        raise orig_exc
    scale = getattr(scaler, "scale_", None)

    mean = np.asarray(mean, dtype=float).reshape(1, -1)
    if scale is None:
        scale = np.ones(mean.shape[1], dtype=float)
    scale = np.asarray(scale, dtype=float).reshape(1, -1)
    scale = np.where(scale == 0, 1.0, scale)

    if x.shape[1] != mean.shape[1]:
        raise ValueError(f"Scaler fallback shape mismatch: X has {x.shape[1]}, scaler has {mean.shape[1]}") from orig_exc

    print(f"[compat] Fallback manual StandardScaler transform ({type(orig_exc).__name__}: {orig_exc})")
    return (x - mean) / scale


def _safe_predict_positive_proba(model, features: np.ndarray, default_prob: float = 0.5) -> float:
    """
    Prefer model.predict_proba. If a LogisticRegression object is partially
    incompatible across sklearn versions, compute sigmoid(coef*x+intercept).
    """
    try:
        probs = model.predict_proba(features)[0]
        if hasattr(model, "classes_") and 1 in model.classes_:
            idx = int(np.where(model.classes_ == 1)[0][0])
            return float(probs[idx])
        return float(probs[-1])
    except Exception as proba_exc:
        # 1) LogisticRegression-compatible manual probability
        if type(model).__name__ == "LogisticRegression" and hasattr(model, "coef_") and hasattr(model, "intercept_"):
            coef = np.asarray(model.coef_, dtype=float)
            intercept = np.asarray(model.intercept_, dtype=float)
            z = float(features @ coef.T + intercept.reshape(1, -1))
            prob = 1.0 / (1.0 + np.exp(-z))
            print("[compat] Fallback manual LogisticRegression probability")
            return float(np.clip(prob, 0.0, 1.0))

        # 2) decision_function -> sigmoid
        try:
            if hasattr(model, "decision_function"):
                score = np.asarray(model.decision_function(features), dtype=float).reshape(-1)[0]
                prob = 1.0 / (1.0 + np.exp(-score))
                print("[compat] Fallback decision_function -> sigmoid probability")
                return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            pass

        # 3) hard class prediction -> pseudo-probability
        try:
            if hasattr(model, "predict"):
                pred = np.asarray(model.predict(features)).reshape(-1)[0]
                prob = 1.0 if int(pred) == 1 else 0.0
                print("[compat] Fallback predict -> pseudo probability")
                return prob
        except Exception:
            pass

        # 4) keep API alive if model object is version-incompatible
        print(f"[compat] Probability fallback to default={default_prob} due to: {type(proba_exc).__name__}: {proba_exc}")
        return float(default_prob)


# ─────────────────────────────────────────────────────────────────────────────
# Heart Disease Predictor  (v2)
# ─────────────────────────────────────────────────────────────────────────────
class HeartRiskPredictor:
    """
    RandomForestClassifier, 20 features.

    Input keys: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
                RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

    20-feature order (ColumnTransformer: num first, then OHE cat):
        [0]  Age  [1] RestingBP  [2] Cholesterol  [3] FastingBS  [4] MaxHR  [5] Oldpeak
        [6]  Sex_F  [7] Sex_M
        [8]  ChestPainType_ASY  [9] ChestPainType_ATA  [10] ChestPainType_NAP  [11] ChestPainType_TA
        [12] RestingECG_LVH  [13] RestingECG_Normal  [14] RestingECG_ST
        [15] ExerciseAngina_N  [16] ExerciseAngina_Y
        [17] ST_Slope_Down  [18] ST_Slope_Flat  [19] ST_Slope_Up
    """

    THRESHOLD   = 0.5
    NUMERICAL   = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

    SEX_CATS    = ["F", "M"]
    CPT_CATS    = ["ASY", "ATA", "NAP", "TA"]
    ECG_CATS    = ["LVH", "Normal", "ST"]
    ANGINA_CATS = ["N", "Y"]
    SLOPE_CATS  = ["Down", "Flat", "Up"]

    def __init__(self, model_path: str, scaler_path: str, columns_path: str):
        for p in [model_path, scaler_path, columns_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Heart model file not found: {p}")
        self.model    = _load_pkl(model_path)
        self.scaler   = _load_pkl(scaler_path)
        self.raw_cols = _load_pkl(columns_path)
        print(f"[HeartRiskPredictor] v2 loaded — {type(self.model).__name__}, "
              f"{self.model.n_features_in_} features, threshold={self.THRESHOLD}")

    def _ohe(self, value, categories: list) -> list:
        return [1 if value == c else 0 for c in categories]

    def _build_features(self, d: dict) -> np.ndarray:
        """Build exact 20-column feature vector matching training pipeline."""
        num_raw    = np.array([[d["Age"], d["RestingBP"], d["Cholesterol"],
                                d["FastingBS"], d["MaxHR"], d["Oldpeak"]]], dtype=float)
        num_scaled = _safe_scale(self.scaler, num_raw)[0].tolist()
        cat_feats  = (
            self._ohe(d["Sex"],            self.SEX_CATS)    +
            self._ohe(d["ChestPainType"],  self.CPT_CATS)    +
            self._ohe(d["RestingECG"],     self.ECG_CATS)    +
            self._ohe(d["ExerciseAngina"], self.ANGINA_CATS) +
            self._ohe(d["ST_Slope"],       self.SLOPE_CATS)
        )
        features = np.array(num_scaled + cat_feats, dtype=float).reshape(1, -1)
        assert features.shape[1] == 20, f"Expected 20 features, got {features.shape[1]}"
        return features

    def predict(self, patient_data: dict) -> dict:
        features = self._build_features(patient_data)
        prob     = _safe_predict_positive_proba(self.model, features)
        band     = _risk_band(prob)
        return {"probability": round(prob, 4), "risk_level": band["level"], **band}


# ─────────────────────────────────────────────────────────────────────────────
# Stroke Risk Predictor  (v3 — recall-first, ADASYN, engineered features)
# ─────────────────────────────────────────────────────────────────────────────
class StrokeRiskPredictor:
    """
    v3 stroke model: LogisticRegression(C=0.1, class_weight=balanced) + ADASYN.
    Trained with engineered features. BMI is DROPPED from this model.
    Threshold = 0.0202 (tuned for 90% recall target).

    Input (raw values — caller provides these):
        age               int/float   e.g. 54
        avg_glucose_level float       e.g. 105.0
        gender            str         "Male" | "Female" | "Other"
        hypertension      int         0 | 1
        heart_disease     int         0 | 1
        ever_married      str         "Yes" | "No"
        work_type         str         "Private"|"Self-employed"|"Govt_job"|"children"|"Never_worked"
        Residence_type    str         "Urban" | "Rural"
        smoking_status    str         "formerly smoked"|"never smoked"|"smokes"|"Unknown"
        (bmi is accepted but ignored — model was trained without it)

    26-feature order (ColumnTransformer: num first, then OHE cat):
        Numeric (5, scaled):
        [0]  age
        [1]  hypertension
        [2]  heart_disease
        [3]  avg_glucose_level
        [4]  age_glucose_interact        (= age * avg_glucose_level, engineered)

        Categorical OHE (21):
        [5]  gender_Female  [6]  gender_Male  [7]  gender_Other
        [8]  ever_married_No  [9]  ever_married_Yes
        [10] work_type_Govt_job  [11] work_type_Never_worked  [12] work_type_Private
        [13] work_type_Self-employed  [14] work_type_children
        [15] Residence_type_Rural  [16] Residence_type_Urban
        [17] smoking_status_Unknown  [18] smoking_status_formerly smoked
        [19] smoking_status_never smoked  [20] smoking_status_smokes
        [21] age_group_<40  [22] age_group_40-60  [23] age_group_>60
        [24] glucose_risk_bin_high_glucose  [25] glucose_risk_bin_normal_glucose
    """

    THRESHOLD     = 0.0202   # Recall-first threshold — 90% recall target (notebook cell 10)

    # Numeric cols in exact training order (must match ColumnTransformer)
    NUMERICAL     = ["age", "hypertension", "heart_disease", "avg_glucose_level", "age_glucose_interact"]

    # OHE category orders (alphabetical — sklearn default)
    GENDER_CATS   = ["Female", "Male", "Other"]
    MARRIED_CATS  = ["No", "Yes"]
    WORK_CATS     = ["Govt_job", "Never_worked", "Private", "Self-employed", "children"]
    RES_CATS      = ["Rural", "Urban"]
    SMOKE_CATS    = ["Unknown", "formerly smoked", "never smoked", "smokes"]
    AGEGROUP_CATS = ["<40", "40-60", ">60"]
    GLUCOSE_CATS  = ["high_glucose", "normal_glucose"]

    def __init__(self, model_path: str, scaler_path: str, columns_path: str):
        for p in [model_path, scaler_path, columns_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Stroke model file not found: {p}")
        self.model    = _load_pkl(model_path)
        self.scaler   = _load_pkl(scaler_path)
        self.raw_cols = _load_pkl(columns_path)
        self._apply_sklearn_compat()
        print(f"[StrokeRiskPredictor] v3 loaded — {type(self.model).__name__}, "
              f"{self.model.n_features_in_} features, threshold={self.THRESHOLD}")

    def _apply_sklearn_compat(self):
        """
        Older sklearn runtimes (e.g. 1.4.x) expect `LogisticRegression.multi_class`
        during predict_proba. Some newer-saved pickles may not contain it.
        """
        if type(self.model).__name__ == "LogisticRegression" and not hasattr(self.model, "multi_class"):
            self.model.multi_class = "auto"
            print("[compat] Added missing LogisticRegression.multi_class='auto'")

    def _ohe(self, value, categories: list) -> list:
        return [1 if value == c else 0 for c in categories]

    def _age_group(self, age: float) -> str:
        if age <= 40:   return "<40"
        elif age <= 60: return "40-60"
        else:           return ">60"

    def _build_features(self, d: dict) -> np.ndarray:
        """Build exact 26-column feature vector matching training pipeline."""
        age  = float(d["age"])
        gluc = float(d["avg_glucose_level"])
        hyp  = int(d["hypertension"])
        hd   = int(d["heart_disease"])

        # Engineered features (must match notebook cell 3 exactly)
        age_glucose_interact = age * gluc
        age_group            = self._age_group(age)
        glucose_risk_bin     = "high_glucose" if gluc >= 126 else "normal_glucose"

        # Scale numeric cols (same order as training ColumnTransformer)
        num_raw    = np.array([[age, hyp, hd, gluc, age_glucose_interact]], dtype=float)
        num_scaled = _safe_scale(self.scaler, num_raw)[0].tolist()

        # OHE categorical cols (alphabetical order — sklearn default)
        cat_feats = (
            self._ohe(d["gender"],         self.GENDER_CATS)   +
            self._ohe(d["ever_married"],   self.MARRIED_CATS)  +
            self._ohe(d["work_type"],      self.WORK_CATS)     +
            self._ohe(d["Residence_type"], self.RES_CATS)      +
            self._ohe(d["smoking_status"], self.SMOKE_CATS)    +
            self._ohe(age_group,           self.AGEGROUP_CATS) +
            self._ohe(glucose_risk_bin,    self.GLUCOSE_CATS)
        )

        features = np.array(num_scaled + cat_feats, dtype=float).reshape(1, -1)
        assert features.shape[1] == 26, f"Expected 26 features, got {features.shape[1]}"
        return features

    def _stroke_risk_band(self, prob: float) -> dict:
        """Threshold-relative banding — stroke model gives raw probs in 1-10% range."""
        if prob >= self.THRESHOLD:
            return {"level": "HIGH",     "color": "#e8192c", "label": "High Risk",     "risk_label": "High Risk"}
        elif prob >= self.THRESHOLD * 0.5:
            return {"level": "MODERATE", "color": "#f59e0b", "label": "Moderate Risk", "risk_label": "Moderate Risk"}
        else:
            return {"level": "LOW",      "color": "#00c97d", "label": "Low Risk",      "risk_label": "Low Risk"}

    def predict(self, patient_data: dict) -> dict:
        features = self._build_features(patient_data)
        prob     = _safe_predict_positive_proba(self.model, features)
        band     = self._stroke_risk_band(prob)
        return {"probability": round(prob, 4), "risk_level": band["level"], **band}
