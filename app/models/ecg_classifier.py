"""
ECG Classification Wrapper
Architecture : CNN + LSTM (1D Conv → LSTM → Sigmoid)
Dataset      : PTB Diagnostic ECG (Normal / Abnormal)
Input        : CSV file — 188 rows × 1 column (187 signal values + 1 label row)
Output       : Normal | Abnormal + probability + waveform base64
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ── Model Architecture — EXACT replica of notebook CNN_LSTM ──────────────────
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        # ── CNN Feature Extractor ─────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(1,   64,  kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),            # 187 → 93

            nn.Conv1d(64,  128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)             # 93 → 46
        )

        # ── LSTM Temporal Modelling ───────────────────────────
        # After 2x MaxPool: seq_len=46, input_size=128
        self.lstm     = nn.LSTM(input_size=128, hidden_size=128,
                                num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)

        # ── Classifier Head ───────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,  1),  nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)              # (B, 128, 46)
        x = x.permute(0, 2, 1)      # (B, 46, 128) ← LSTM needs (batch, seq, features)
        _, (h_n, _) = self.lstm(x)  # h_n: (1, B, 128) — final hidden state
        x = h_n.squeeze(0)          # (B, 128)
        x = self.dropout1(x)
        x = self.classifier(x)      # (B, 1)
        return x


def _make_waveform_chart(signal: np.ndarray, label: str, confidence: float) -> str:
    """Render ECG waveform as a styled dark chart. Returns base64 PNG."""
    color = "#e74c3c" if label == "ABNORMAL" else "#2ecc71"

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    ax.plot(signal, color=color, linewidth=1.5, alpha=0.95)
    ax.fill_between(range(len(signal)), signal, alpha=0.12, color=color)

    # R-peak marker
    peak_idx = int(np.argmax(signal))
    ax.annotate(
        f"R-peak: {signal[peak_idx]:.3f}",
        xy=(peak_idx, signal[peak_idx]),
        xytext=(peak_idx + 10, signal[peak_idx] + 0.05),
        color="white", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="white", lw=0.8)
    )

    # Grid lines
    ax.grid(color="#1f2937", linewidth=0.5, alpha=0.8)
    ax.set_title(
        f"{label}  •  Confidence: {confidence*100:.1f}%",
        color=color, fontsize=11, fontweight="bold", pad=8
    )
    ax.set_xlabel("Time Steps (0 – 186)", color="#6b7280", fontsize=8)
    ax.set_ylabel("Amplitude",            color="#6b7280", fontsize=8)
    ax.tick_params(colors="#6b7280", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1f2937")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class ECGClassifier:
    """Loads once at startup. Call predict(csv_bytes) for inference."""

    THRESHOLD = 0.5

    def __init__(self, weights_path: str, device: str = None):
        self.device = torch.device(
            device if device else (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.model = CNN_LSTM()
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[ECGClassifier] Loaded on {self.device}")

    def _load_weights(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"ECG weights not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        # Handle both raw state_dict and wrapped checkpoint
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state, strict=True)

    def _parse_csv(self, csv_bytes: bytes) -> np.ndarray:
        """
        CSV format: 188 rows × 1 column.
        Rows 0–186 = signal. Row 187 = label (dropped).
        """
        df = pd.read_csv(io.BytesIO(csv_bytes), header=None)
        values = df.iloc[:, 0].values.astype(np.float32)
        signal = values[:187]
        return signal

    def predict(self, csv_bytes: bytes) -> dict:
        signal = self._parse_csv(csv_bytes)

        # Shape: (1, 1, 187)
        tensor = torch.tensor(signal).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = float(self.model(tensor).item())

        label      = "ABNORMAL" if prob >= self.THRESHOLD else "NORMAL"
        confidence = prob if label == "ABNORMAL" else (1.0 - prob)
        color      = "#e8192c" if label == "ABNORMAL" else "#00c97d"

        waveform_b64 = _make_waveform_chart(signal, label, confidence)

        return {
            "label":        label,
            "probability":  round(prob, 4),
            "confidence":   round(confidence, 4),
            "status_color": color,
            "waveform_b64": waveform_b64,
            "signal":       signal.tolist()
        }
