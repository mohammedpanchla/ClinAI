"""
ClinAI — FastAPI Backend
────────────────────────
Single endpoint: POST /api/analyze
Accepts multipart form data, runs pipeline, returns JSON + PDF.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional
import uvicorn

from app.services.pipeline import run_diagnostic_pipeline
from app.services.report_generator import generate_report

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClinAI Diagnostic System",
    description="End-to-end clinical AI: Brain MRI · ECG · Cardiovascular Risk",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve reports and static files
Path("reports").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/static",  StaticFiles(directory="static"),  name="static")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "system": "ClinAI v1.0"}


# ── Main analysis endpoint ─────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    # ── Patient info ──────────────────────────────────────────────────────
    patient_name:    str = Form(...),
    patient_age:     int = Form(...),
    patient_gender:  str = Form(...),

    # ── Files ─────────────────────────────────────────────────────────────
    detection_image: UploadFile = File(...),
    ecg_csv:         UploadFile = File(...),
    segmentation_image: Optional[UploadFile] = File(None),

    # ── Heart / clinical ─────────────────────────────────────────────────
    resting_bp:      float = Form(...),
    cholesterol:     float = Form(...),
    max_hr:          float = Form(...),
    fasting_bs:      int   = Form(...),
    oldpeak:         float = Form(...),
    chest_pain_type: str   = Form(...),
    resting_ecg:     str   = Form(...),
    exercise_angina: int   = Form(...),
    st_slope:        str   = Form(...),

    # ── Stroke / clinical ─────────────────────────────────────────────────
    bmi:                   float = Form(...),
    avg_glucose_level:     float = Form(...),
    hypertension:          int   = Form(...),
    heart_disease_history: int   = Form(...),
    ever_married:          int   = Form(...),
    is_urban:              int   = Form(...),
    work_type:             str   = Form(...),
    smoking_status:        str   = Form(...),
):
    try:
        # Read file bytes
        detection_bytes  = await detection_image.read()
        ecg_bytes        = await ecg_csv.read()
        det_ext          = detection_image.filename.rsplit(".", 1)[-1]

        seg_bytes = None
        seg_ext   = "tif"
        if segmentation_image and segmentation_image.filename:
            seg_bytes = await segmentation_image.read()
            seg_ext   = segmentation_image.filename.rsplit(".", 1)[-1]

        # Run the full pipeline
        result = run_diagnostic_pipeline(
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            detection_image_bytes=detection_bytes,
            detection_image_ext=det_ext,
            ecg_csv_bytes=ecg_bytes,
            resting_bp=resting_bp,
            cholesterol=cholesterol,
            max_hr=max_hr,
            fasting_bs=fasting_bs,
            oldpeak=oldpeak,
            chest_pain_type=chest_pain_type,
            resting_ecg=resting_ecg,
            exercise_angina=exercise_angina,
            st_slope=st_slope,
            bmi=bmi,
            avg_glucose_level=avg_glucose_level,
            hypertension=hypertension,
            heart_disease_history=heart_disease_history,
            ever_married=ever_married,
            is_urban=is_urban,
            work_type=work_type,
            smoking_status=smoking_status,
            segmentation_image_bytes=seg_bytes,
            segmentation_image_ext=seg_ext,
        )

        # Generate PDF
        report = generate_report(result, detection_bytes)

        return JSONResponse({
            "success":   True,
            "result":    result,
            "pdf_url":   report["pdf_url"],
            "report_id": result["report_id"],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve the frontend ─────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
