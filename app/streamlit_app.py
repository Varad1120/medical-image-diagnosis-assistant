"""
Streamlit Dashboard for Medical Image Diagnosis Assistant.
Run with: streamlit run app/streamlit_app.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import get_inference_transforms
from src.models.efficientnet import EfficientNetDiagnosisModel
from src.utils.gradcam import GradCAM

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🩺 Pneumonia Detection AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0f1117; }

    .diagnosis-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
    }

    .result-normal {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }

    .result-pneumonia {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }

    .metric-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #334155;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }

    h1 { color: #f1f5f9 !important; }
    h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/config.yaml")
MODEL_PATH  = Path("models/best_model.pth")
CLASSES     = ["NORMAL", "PNEUMONIA"]
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load Model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model = EfficientNetDiagnosisModel(
        num_classes=config["model"]["num_classes"],
        pretrained=False,
        dropout=0.0,
    )

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, config, True
    else:
        return None, config, False


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(model, image: Image.Image, config: dict):
    transform = get_inference_transforms(config["data"]["image_size"])
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx    = probs.argmax().item()
    prediction  = CLASSES[pred_idx]
    confidence  = probs[pred_idx].item()
    prob_normal = probs[0].item()
    prob_pneumo = probs[1].item()

    return prediction, confidence, prob_normal, prob_pneumo, tensor


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=True)
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.99, 0.75, 0.01)
    st.divider()
    st.markdown("### 📋 About")
    st.markdown("""
    This AI assistant detects **Pneumonia** from chest X-ray images using:
    - **EfficientNet-B3** backbone
    - **Transfer Learning** on ImageNet
    - **Grad-CAM** for visual explainability

    > ⚠️ For **research use only** — not a clinical tool.
    """)
    st.divider()
    st.markdown(f"**Device:** `{device}`")


# ── Main Header / Landing Page ─────────────────────────────────────────────────
st.markdown("# 🩺 Medical Image Diagnosis Assistant")
st.markdown("### AI-Powered Pneumonia Detection from Chest X-Rays")

st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 28px;
    margin: 16px 0 24px 0;
    font-family: 'Inter', sans-serif;
">
    <table style="width:100%; border-collapse:collapse; color:#e2e8f0;">
        <tr>
            <td style="padding:6px 0; font-size:1.05em;">
                📌 <strong style="color:#94a3b8;">Project Topic</strong>
            </td>
            <td style="padding:6px 0; font-size:1.05em; color:#f1f5f9;">
                AI-Powered Medical Image Diagnosis — Pneumonia Detection from Chest X-Rays
            </td>
        </tr>
        <tr>
            <td style="padding:6px 0; font-size:1.05em;">
                👤 <strong style="color:#94a3b8;">Full Name</strong>
            </td>
            <td style="padding:6px 0; font-size:1.05em; color:#f1f5f9;">
                Varad Sachin Kale
            </td>
        </tr>
        <tr>
            <td style="padding:6px 0; font-size:1.05em;">
                📧 <strong style="color:#94a3b8;">Registered Email</strong>
            </td>
            <td style="padding:6px 0; font-size:1.05em; color:#f1f5f9;">
                varadk1120@gmail.com
            </td>
        </tr>
        <tr>
            <td style="padding:6px 0; font-size:1.05em;">
                🏫 <strong style="color:#94a3b8;">Program</strong>
            </td>
            <td style="padding:6px 0; font-size:1.05em; color:#f1f5f9;">
                SparkIIT — ML with Python
            </td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load Model ─────────────────────────────────────────────────────────────────
model, config, model_loaded = load_model()

if not model_loaded:
    st.warning(
        "⚠️ **No trained model found.** Please train the model first:\n\n"
        "```bash\npython src/utils/trainer.py --config configs/config.yaml\n```"
    )
    st.stop()

st.success(f"✅ Model loaded successfully — running on **{device}**")

# ── Upload Section ─────────────────────────────────────────────────────────────
st.markdown("## 📤 Upload Chest X-Ray Image")
uploaded_file = st.file_uploader(
    "Drag and drop or click to upload",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray image (JPEG or PNG)"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🔬 Original X-Ray")
        st.image(image, use_container_width=True, caption=f"Uploaded: {uploaded_file.name}")

    # Run prediction
    with st.spinner("🤖 Analyzing X-Ray..."):
        prediction, confidence, prob_normal, prob_pneumo, tensor = predict(
            model, image, config
        )

    is_pneumonia = prediction == "PNEUMONIA"
    is_confident = confidence >= confidence_threshold

    with col2:
        st.markdown("### 📊 Diagnosis Result")

        if is_pneumonia:
            st.markdown(f"""
            <div class="result-pneumonia">
                <h2 style="color:#fca5a5; margin:0">⚠️ PNEUMONIA DETECTED</h2>
                <p style="color:#fecaca; font-size:1.2em; margin:8px 0">
                    Confidence: <strong>{confidence*100:.1f}%</strong>
                </p>
                {"<p style='color:#fca5a5'>⚠️ Below confidence threshold — treat with caution</p>" if not is_confident else ""}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-normal">
                <h2 style="color:#6ee7b7; margin:0">✅ NORMAL</h2>
                <p style="color:#a7f3d0; font-size:1.2em; margin:8px 0">
                    Confidence: <strong>{confidence*100:.1f}%</strong>
                </p>
                {"<p style='color:#6ee7b7'>⚠️ Below confidence threshold — treat with caution</p>" if not is_confident else ""}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Probability Breakdown")

        st.markdown(f"🟢 **Normal** — {prob_normal*100:.1f}%")
        st.progress(prob_normal)

        st.markdown(f"🔴 **Pneumonia** — {prob_pneumo*100:.1f}%")
        st.progress(prob_pneumo)

    # Grad-CAM
    if show_gradcam:
        st.markdown("---")
        st.markdown("## 🔥 Grad-CAM Explainability")
        st.caption("Highlighted regions show where the model focused to make its decision.")

        try:
            target_layer = model.backbone.blocks[-1]
            gradcam = GradCAM(model, target_layer)
            cam = gradcam.generate(tensor.requires_grad_(True))

            img_np = np.array(image.resize((224, 224)))
            overlay = gradcam.overlay_on_image(img_np, cam)

            gcol1, gcol2, gcol3 = st.columns(3)
            with gcol1:
                st.image(image, caption="Original", use_container_width=True)
            with gcol2:
                st.image(cam, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
            with gcol3:
                st.image(overlay, caption="Overlay", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM visualization skipped: {e}")

    # Detailed metrics card
    st.markdown("---")
    st.markdown("## 📋 Prediction Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction", prediction)
    m2.metric("Confidence", f"{confidence*100:.1f}%")
    m3.metric("Normal Prob.", f"{prob_normal*100:.1f}%")
    m4.metric("Pneumonia Prob.", f"{prob_pneumo*100:.1f}%")

    st.caption("⚠️ Disclaimer: This tool is for educational and research purposes only. It is not intended for clinical diagnosis.")

else:
    # Demo placeholder
    st.info("👆 Upload a chest X-ray image above to get started.")

    st.markdown("### 💡 Example X-Rays to Try")
    st.markdown("""
    Download sample images from the Kaggle dataset and upload them here:
    - **Normal**: Clear lung fields, no infiltrates
    - **Pneumonia**: Opacification, consolidation visible in lungs

    📥 Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    """)
