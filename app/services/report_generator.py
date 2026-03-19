"""
Report Generator Service
────────────────────────
Takes the unified result dict from pipeline.py,
renders the Jinja2 HTML template, and converts to PDF via WeasyPrint.

Also prepares base64-encoded images for embedding.
"""

import os
import base64
import io
import uuid
from pathlib import Path
from datetime import datetime

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML as WeasyHTML

from PIL import Image


# ── Template environment ───────────────────────────────────────────────────────
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
REPORTS_DIR  = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))


def _image_bytes_to_b64(image_bytes: bytes, max_size: int = 400) -> str:
    """
    Convert raw image bytes to base64 PNG string.
    Resizes to max_size on longest side to keep PDF lean.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Downscale if needed
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _normalize_b64_image(image_b64: str, max_width: int, max_height: int) -> str:
    """
    Decode an already-generated base64 image, constrain it to a predictable
    size, and re-encode as PNG. This keeps WeasyPrint image boxes stable.
    """
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    img.thumbnail((max_width, max_height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _prepare_template_context(result: dict, detection_image_bytes: bytes, det_ext: str) -> dict:
    context = dict(result)

    segmentation = result.get("segmentation")
    if segmentation:
        segmentation = dict(segmentation)
        size_map = {
            "seg_image_b64": (480, 480),
            "mask_b64": (480, 480),
            "overlay_b64": (480, 480),
            "heatmap_b64": (900, 420),
        }
        for key, (max_width, max_height) in size_map.items():
            if segmentation.get(key):
                segmentation[key] = _normalize_b64_image(segmentation[key], max_width, max_height)
        context["segmentation"] = segmentation

    context["detection_image_b64"] = _image_bytes_to_b64(detection_image_bytes, max_size=480)
    context["detection_ext"] = det_ext
    return context


def generate_report(
    result: dict,
    detection_image_bytes: bytes,
) -> dict:
    """
    Args:
        result               : unified dict from pipeline.run_diagnostic_pipeline()
        detection_image_bytes: raw bytes of the detection MRI image

    Returns:
        {
            "pdf_path"   : str  (absolute path to saved PDF),
            "pdf_url"    : str  (relative URL for download),
            "html"       : str  (rendered HTML — for web preview),
            "report_id"  : str
        }
    """

    # ── Prepare template context ───────────────────────────────────────────────
    # Detect file extension from bytes (PNG magic bytes check)
    if detection_image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        det_ext = "png"
    elif detection_image_bytes[:3] == b'\xff\xd8\xff':
        det_ext = "jpg"
    elif detection_image_bytes[:4] in (b'II*\x00', b'MM\x00*'):
        det_ext = "tif"
    else:
        det_ext = "img"

    context = _prepare_template_context(result, detection_image_bytes, det_ext)

    # ── Render HTML ────────────────────────────────────────────────────────────
    template  = jinja_env.get_template("report.html")
    html_str  = template.render(**context)

    # ── Convert to PDF ─────────────────────────────────────────────────────────
    report_id  = result["report_id"]
    filename   = f"ClinAI_Report_{report_id}.pdf"
    pdf_path   = REPORTS_DIR / filename

    try:
        WeasyHTML(string=html_str, base_url=str(TEMPLATE_DIR)).write_pdf(str(pdf_path))
    except AttributeError as e:
        if "super" in str(e) and "transform" in str(e):
            raise RuntimeError(
                "PDF generation dependency mismatch: WeasyPrint 62.x requires "
                "pydyf<0.11.0. Pin pydyf to 0.10.x and rebuild the image."
            ) from e
        raise

    return {
        "pdf_path":  str(pdf_path),
        "pdf_url":   f"/reports/{filename}",
        "html":      html_str,
        "report_id": report_id,
    }
