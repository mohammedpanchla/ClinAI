"""
Tumor Segmentation Wrapper
Architecture : U-Net + VGG16 encoder (segmentation_models_pytorch)
Dataset      : LGG Kaggle (TCGA-style TIFF scans)
Input        : TIF / PNG — 3-channel, ImageNet normalized via albumentations
Output       : binary mask + heatmap + overlay (all base64 PNG)
"""

import torch
import numpy as np
import tifffile
import cv2
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ── Validation transform ───────────────────────────────────────────────────────
VAL_TRANSFORM = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def _build_model() -> torch.nn.Module:
    return smp.Unet(
        encoder_name="vgg16",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )


def _to_b64(arr_bgr: np.ndarray) -> str:
    """Encode a BGR numpy image to base64 PNG string."""
    _, buf = cv2.imencode(".png", arr_bgr)
    return base64.b64encode(buf).decode("utf-8")


class TumorSegmentor:

    def __init__(self, weights_path: str, device: str = None):
        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.model = _build_model()
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[TumorSegmentor] Loaded on {self.device}")

    def _load_weights(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Segmentation weights not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)

    def _load_image(self, image_bytes: bytes, ext: str) -> np.ndarray:
        """Load TIFF or PNG/JPG → uint8 RGB (H, W, 3)."""
        ext = ext.lower().strip(".")
        if ext in ("tif", "tiff"):
            arr = tifffile.imread(io.BytesIO(image_bytes))
        else:
            arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

        if arr.dtype != np.uint8:
            arr = (arr / arr.max() * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        return arr

    def _make_mask_b64(self, mask: np.ndarray) -> str:
        """
        Render the binary mask as a clean dark-red blob on white background.
        Returns base64 PNG.
        """
        h, w = mask.shape
        # White background
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 245

        # Dark red tumor region
        canvas[mask == 1] = [100, 20, 20]   # dark red in BGR

        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (80, 10, 10), 1)

        return _to_b64(canvas)

    def _make_overlay_b64(self, original_rgb: np.ndarray, mask: np.ndarray) -> str:
        """Red tumor overlay blended onto the original MRI. Returns base64 PNG."""
        h, w = original_rgb.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = original_rgb.copy()
        overlay[mask_resized == 1] = [220, 50, 50]

        blended = cv2.addWeighted(original_rgb, 0.6, overlay, 0.4, 0)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 80, 80), 2)

        return _to_b64(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    def _make_heatmap_b64(self, prob_map: np.ndarray) -> str:
        """
        Render the sigmoid probability map as a full-size HOT heatmap with colorbar.
        Uses matplotlib Agg backend (non-interactive, safe for server/WeasyPrint).
        Output is upscaled to 800x600 so it fills the PDF container properly.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(8, 5), dpi=100, facecolor="#0a0a1a")
        ax.set_facecolor("#0a0a1a")

        # Upscale prob_map for better resolution in PDF
        prob_up = cv2.resize(prob_map, (512, 512), interpolation=cv2.INTER_LINEAR)

        im = ax.imshow(prob_up, cmap="hot", vmin=0.0, vmax=1.0,
                       aspect="auto", extent=[0, 512, 0, 512])

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.set_label("Confidence", color="white", fontsize=8)

        ax.set_title("Confidence Map · Sigmoid Probability Map",
                     color="white", fontsize=10, pad=8, fontweight="bold")
        ax.axis("off")

        fig.tight_layout(pad=0.8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100,
                    bbox_inches="tight", facecolor="#0a0a1a")
        plt.close(fig)
        buf.seek(0)
        data = buf.read()
        buf.close()
        return base64.b64encode(data).decode("utf-8")

    def _make_seg_input_b64(self, original_rgb: np.ndarray) -> str:
        """Convert the raw segmentation input image to base64 PNG for the report."""
        h, w = original_rgb.shape[:2]
        # Resize to a reasonable size for PDF embedding
        target = 400
        if max(h, w) > target:
            ratio = target / max(h, w)
            original_rgb = cv2.resize(original_rgb, (int(w * ratio), int(h * ratio)),
                                      interpolation=cv2.INTER_AREA)
        return _to_b64(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

    def predict(self, image_bytes: bytes, file_ext: str = "tif") -> dict:
        original = self._load_image(image_bytes, file_ext)

        aug    = VAL_TRANSFORM(image=original, mask=np.zeros(original.shape[:2], dtype=np.float32))
        tensor = aug["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits   = self.model(tensor)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()  # (256, 256)

        mask = (prob_map > 0.5).astype(np.uint8)

        total_pixels = mask.size
        tumor_pixels = int(mask.sum())
        coverage     = round((tumor_pixels / total_pixels) * 100, 2)
        detected     = tumor_pixels > 50

        # ── Generate all four images ───────────────────────────────────────────
        seg_image_b64 = self._make_seg_input_b64(original)   # raw input image
        mask_b64      = self._make_mask_b64(mask)            # always include a mask preview
        overlay_b64   = self._make_overlay_b64(original, mask)
        heatmap_b64   = self._make_heatmap_b64(prob_map)     # always generate

        return {
            "tumor_detected":   detected,
            "coverage_percent": coverage,
            "mask_pixel_count": tumor_pixels,
            "iou_note":         "Segmentation model: Val IoU 71.1%, Dice 78.2% (LGG dataset)",
            "seg_image_b64":    seg_image_b64,  # ← raw segmentation input image
            "mask_b64":         mask_b64,        # ← binary mask
            "overlay_b64":      overlay_b64,     # ← red overlay on MRI
            "heatmap_b64":      heatmap_b64,     # ← sigmoid probability heatmap
        }


# ── Module-level helper (avoids self reference issue) ─────────────────────────
def _make_mask_b64_fn(mask: np.ndarray) -> str:
    h, w = mask.shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 245
    canvas[mask == 1] = [100, 20, 20]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (80, 10, 10), 1)
    _, buf = cv2.imencode(".png", canvas)
    return base64.b64encode(buf).decode("utf-8")
