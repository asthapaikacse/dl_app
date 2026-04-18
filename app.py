import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
import gdown

# Add src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import ImprovedStrokeViT

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Brain Stroke AI", page_icon="🧠", layout="centered")

MODEL_URL = "https://drive.google.com/uc?id=1AbDBoYmzfS9sMz4iSVz2SlsPPqTaWo9k"
MODEL_PATH = "improved_vit_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DOWNLOAD MODEL ----------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... (first time only)")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

download_model()

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00E5FF;
}
.result-box {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    font-size: 22px;
}
.normal {
    background-color: #e8f5e9;
    color: #2e7d32;
}
.stroke {
    background-color: #ffebee;
    color: #c62828;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
   checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False   
    )

    model = ImprovedStrokeViT(
        num_labels=2,
        unfreeze_layers=6,
        dropout=0.1
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold = checkpoint.get("optimal_threshold", 0.5)

    return model, threshold

model, threshold = load_model()

# ---------------- TRANSFORM ----------------
def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# ---------------- ATTENTION ----------------
def get_attention_map(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor, return_features=True)
        _, _, attn_weights = outputs

    attn = attn_weights.squeeze().cpu().numpy()
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    attn = np.resize(attn, (14, 14))
    attn = np.kron(attn, np.ones((16, 16)))

    return attn

# ---------------- PREDICT ----------------
def predict(model, image):
    transform = get_transform()
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = torch.softmax(outputs, dim=1)

        stroke_prob = probs[0, 1].item()
        normal_prob = probs[0, 0].item()

        prediction = 1 if stroke_prob >= threshold else 0
        confidence = max(stroke_prob, normal_prob)

    return prediction, stroke_prob, normal_prob, confidence, img

# ---------------- UI ----------------
st.markdown('<div class="main-title">🧠 Brain Stroke Detection</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload MRI/CT Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_column_width=True)

    pred, stroke_prob, normal_prob, conf, img_tensor = predict(model, image)

    with col2:
        if pred == 1:
            st.markdown(
                f'<div class="result-box stroke">⚠️ STROKE<br>{conf*100:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box normal">✅ NORMAL<br>{conf*100:.2f}%</div>',
                unsafe_allow_html=True
            )

        st.write(f"Stroke: {stroke_prob:.4f}")
        st.write(f"Normal: {normal_prob:.4f}")

    # Attention map
    st.subheader("🔍 Model Attention")

    attn_map = get_attention_map(model, img_tensor)

    img_np = np.array(image.resize((224, 224)))
    heatmap = plt.cm.jet(attn_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = 0.6 * img_np + 0.4 * heatmap

    st.image(overlay.astype(np.uint8), use_column_width=True)
