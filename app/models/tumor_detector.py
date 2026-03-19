"""
Tumor Detection Wrapper
Architecture : VGG16 + Custom Classifier Head
Dataset      : BraTS (Glioma, Meningioma, Pituitary, No Tumor)
Input        : JPG / PNG — resized to 224×224, ImageNet normalized
Output       : class label + confidence score
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
from pathlib import Path


# ── Class labels (same order as training folder sort) ─────────────────────────
TUMOR_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

SEVERITY_MAP = {
    "glioma":     {"severity": "HIGH",     "color": "#e74c3c"},
    "meningioma": {"severity": "MODERATE", "color": "#e67e22"},
    "pituitary":  {"severity": "MODERATE", "color": "#e67e22"},
    "notumor":    {"severity": "NONE",     "color": "#2ecc71"},
}

# ── ImageNet normalization (matches training) ──────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def _build_model(num_classes: int = 4) -> nn.Module:
    """Rebuild VGG16 with the exact classifier head used during training."""
    model = models.vgg16(weights=None)

    # Freeze features (not needed at inference, but keeps architecture identical)
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model


class TumorDetector:
    """
    Loads once at startup, stays in memory.
    Call predict(image_bytes) for inference.
    """

    def __init__(self, weights_path: str, device: str = None):
        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.model = _build_model(num_classes=4)
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[TumorDetector] Loaded on {self.device}")

    def _load_weights(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Tumor detector weights not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        # Handle both raw state_dict and checkpoint dicts
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)

    def predict(self, image_bytes: bytes) -> dict:
        """
        Args:
            image_bytes: raw bytes of a JPG or PNG file

        Returns:
            {
                "tumor_class"  : "glioma" | "meningioma" | "pituitary" | "notumor",
                "confidence"   : 0.0 – 1.0,
                "severity"     : "HIGH" | "MODERATE" | "NONE",
                "severity_color": hex color string,
                "all_probs"    : {class: prob, ...}   # for report bar chart
            }
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0).to(self.device)   # (1, 3, 224, 224)

        with torch.no_grad():
            logits = self.model(tensor)                           # (1, 4)
            probs  = torch.softmax(logits, dim=1).squeeze()      # (4,)

        probs_np   = probs.cpu().numpy()
        pred_idx   = int(probs_np.argmax())
        pred_class = TUMOR_CLASSES[pred_idx]
        confidence = float(probs_np[pred_idx])

        return {
            "tumor_class":    pred_class,
            "confidence":     round(confidence, 4),
            "severity":       SEVERITY_MAP[pred_class]["severity"],
            "severity_color": SEVERITY_MAP[pred_class]["color"],
            "all_probs": {
                cls: round(float(probs_np[i]), 4)
                for i, cls in enumerate(TUMOR_CLASSES)
            }
        }
