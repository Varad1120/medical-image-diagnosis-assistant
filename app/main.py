"""
FastAPI application for Medical Image Diagnosis inference.
"""

import io
from pathlib import Path
from typing import Dict

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.data.transforms import get_inference_transforms
from src.models.efficientnet import EfficientNetDiagnosisModel

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🩺 Medical Image Diagnosis API",
    description="AI-powered Pneumonia Detection from Chest X-Ray images using EfficientNet-B3 with Grad-CAM explainability.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Config & Model ────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/config.yaml")
MODEL_PATH = Path("models/best_model.pth")

CLASSES = ["NORMAL", "PNEUMONIA"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
transform = None


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global model, transform

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    transform = get_inference_transforms(config["data"]["image_size"])

    if MODEL_PATH.exists():
        model = EfficientNetDiagnosisModel(
            num_classes=config["model"]["num_classes"],
            pretrained=False,
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  No trained model found at {MODEL_PATH}. Train a model first.")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root() -> Dict:
    return {
        "message": "🩺 Medical Image Diagnosis API is running",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.get("/health", tags=["Health"])
async def health() -> Dict:
    return {"status": "healthy", "model_ready": model is not None}


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Upload a chest X-ray image and receive a diagnosis prediction.

    Returns:
        - prediction: 'NORMAL' or 'PNEUMONIA'
        - confidence: probability score (0-1)
        - probabilities: scores for each class
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train first.")

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    prediction = CLASSES[pred_idx]
    confidence = probs[pred_idx].item()

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)},
        "filename": file.filename,
    }
