import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image
from utils.feature_extractor import extract_features
from utils.disease_info import get_disease_info

st.set_page_config(
    page_title="CropGuard AI — Crop Disease Detector",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: #f0fdf4;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 40%, #166534 100%);
    border-radius: 24px;
    padding: 50px 40px 40px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(22,163,74,0.25);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
    transform: rotate(-15deg);
}
.hero-title {
    font-size: 2.8em;
    font-weight: 800;
    color: white;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
}
.hero-sub {
    color: #bbf7d0;
    font-size: 1em;
    font-weight: 400;
    margin-bottom: 30px;
}
.stats-row {
    display: flex;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 50px;
    padding: 10px 24px;
    color: white;
    backdrop-filter: blur(10px);
}
.stat-num {
    font-size: 1.4em;
    font-weight: 700;
    display: block;
}
.stat-lbl {
    font-size: 0.7em;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #bbf7d0;
}

/* ── CROP TAGS ── */
.crop-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.crop-tag {
    background: white;
    border: 1.5px solid #86efac;
    border-radius: 50px;
    padding: 6px 18px;
    font-size: 0.85em;
    font-weight: 500;
    color: #15803d;
    box-shadow: 0 2px 8px rgba(22,163,74,0.1);
}

/* ── PANELS ── */
.panel {
    background: white;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 1px solid #dcfce7;
    height: 100%;
}
.panel-title {
    font-size: 1em;
    font-weight: 600;
    color: #15803d;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── UPLOAD BOX ── */
.upload-hint {
    background: #f0fdf4;
    border: 2px dashed #86efac;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    color: #16a34a;
    font-size: 0.9em;
    margin-bottom: 16px;
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #6b7280;
}
.empty-icon {
    font-size: 3.5em;
    margin-bottom: 12px;
}
.empty-title {
    font-size: 1em;
    font-weight: 600;
    color: #374151;
    margin-bottom: 6px;
}
.empty-sub {
    font-size: 0.85em;
    color: #9ca3af;
}

/* ── DISEASE RESULT ── */
.result-disease {
    background: linear-gradient(135deg, #fef2f2, #fff);
    border: 1.5px solid #fca5a5;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}
.result-healthy {
    background: linear-gradient(135deg, #f0fdf4, #fff);
    border: 1.5px solid #86efac;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}
.result-icon {
    font-size: 2.2em;
    margin-bottom: 8px;
}
.result-name {
    font-size: 1.25em;
    font-weight: 700;
    color: #111827;
    margin-bottom: 4px;
}
.result-conf {
    font-size: 0.85em;
    color: #6b7280;
    font-weight: 500;
}

/* ── CONFIDENCE BAR ── */
.conf-wrap {
    background: #f3f4f6;
    border-radius: 999px;
    height: 8px;
    margin: 12px 0 20px;
    overflow: hidden;
}
.conf-fill-high { background: linear-gradient(90deg, #16a34a, #4ade80); }
.conf-fill-mid  { background: linear-gradient(90deg, #d97706, #fbbf24); }
.conf-fill-low  { background: linear-gradient(90deg, #dc2626, #f87171); }

/* ── INFO CARDS ── */
.info-card {
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.info-card-cause {
    background: #fff1f2;
    border-left: 4px solid #f43f5e;
}
.info-card-symptom {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
}
.info-card-label {
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 5px;
    color: #6b7280;
}
.info-card-text {
    font-size: 0.9em;
    color: #374151;
    line-height: 1.55;
}

/* ── LISTS ── */
.tip-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 7px;
    font-size: 0.88em;
    color: #374151;
    line-height: 1.5;
}
.tip-treatment {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
}
.tip-prevention {
    background: #fefce8;
    border: 1px solid #fde68a;
}
.tip-dot-t { color: #16a34a; font-size: 1.1em; flex-shrink: 0; margin-top: 1px; }
.tip-dot-p { color: #d97706; font-size: 1.1em; flex-shrink: 0; margin-top: 1px; }

.section-label {
    font-size: 0.85em;
    font-weight: 700;
    color: #374151;
    margin: 16px 0 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── FOOTER ── */
.footer {
    background: white;
    border-radius: 14px;
    padding: 18px 24px;
    text-align: center;
    margin-top: 24px;
    border: 1px solid #dcfce7;
    font-size: 0.85em;
    color: #6b7280;
}
.footer a { color: #16a34a; text-decoration: none; font-weight: 500; }
.footer a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model/crop_disease_model.pkl"
    label_path = "model/label_encoder.pkl"
    if os.path.exists(model_path) and os.path.exists(label_path):
        return joblib.load(model_path), joblib.load(label_path)
    return None, None

model, label_encoder = load_model()

# ── HERO ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🌿 CropGuard AI</div>
  <div class="hero-sub">AI-powered crop disease detection for Indian farmers — instant diagnosis, expert treatment advice</div>
  <div class="stats-row">
    <div class="stat-pill"><span class="stat-num">91%</span><span class="stat-lbl">Accuracy</span></div>
    <div class="stat-pill"><span class="stat-num">15</span><span class="stat-lbl">Diseases</span></div>
    <div class="stat-pill"><span class="stat-num">50K+</span><span class="stat-lbl">Images Trained</span></div>
    <div class="stat-pill"><span class="stat-num">&lt;2s</span><span class="stat-lbl">Detection Speed</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Crop Tags ───────────────────────────────────────────────────
st.markdown("""
<div class="crop-row">
  <span class="crop-tag">🍅 Tomato</span>
  <span class="crop-tag">🥔 Potato</span>
  <span class="crop-tag">🌶️ Pepper</span>
</div>
""", unsafe_allow_html=True)

# ── Two Column Layout ───────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📷 Upload Leaf Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">📌 Use a clear, well-lit photo of a single leaf for best accuracy</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=380, caption="Your uploaded leaf")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔬 Analysis Result</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🔬</div>
          <div class="empty-title">No image uploaded yet</div>
          <div class="empty-sub">Upload a leaf photo on the left to get instant AI diagnosis with treatment advice</div>
        </div>
        """, unsafe_allow_html=True)
    elif model is None:
        st.warning("⚠️ Model not found. Please run train_model.py first.")
    else:
        with st.spinner("🔬 Analyzing your leaf..."):
            img_array = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            features = extract_features(img_bgr).reshape(1, -1)
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            confidence = int(max(proba) * 100)
            disease_name = label_encoder.inverse_transform([pred])[0]
            display_name = disease_name.replace("___", " — ").replace("_", " ")
            is_healthy = "healthy" in disease_name.lower()
            info = get_disease_info(disease_name)

        # Result badge
        if is_healthy:
            st.markdown(f"""
            <div class="result-healthy">
              <div class="result-icon">✅</div>
              <div class="result-name">{display_name}</div>
              <div class="result-conf">Confidence: {confidence}%</div>
            </div>
            """, unsafe_allow_html=True)
            fill_class = "conf-fill-high"
        else:
            st.markdown(f"""
            <div class="result-disease">
              <div class="result-icon">🦠</div>
              <div class="result-name">{display_name}</div>
              <div class="result-conf">Confidence: {confidence}%</div>
            </div>
            """, unsafe_allow_html=True)
            fill_class = "conf-fill-high" if confidence >= 70 else "conf-fill-mid" if confidence >= 40 else "conf-fill-low"

        # Confidence bar
        st.markdown(f"""
        <div class="conf-wrap">
          <div class="{fill_class}" style="width:{confidence}%; height:8px; border-radius:999px;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Cause & Symptoms
        st.markdown(f"""
        <div class="info-card info-card-cause">
          <div class="info-card-label">🧬 Cause</div>
          <div class="info-card-text">{info['cause']}</div>
        </div>
        <div class="info-card info-card-symptom">
          <div class="info-card-label">🔍 Symptoms</div>
          <div class="info-card-text">{info['symptoms']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Treatment
        st.markdown('<div class="section-label">💊 Treatment</div>', unsafe_allow_html=True)
        for tip in info['treatment']:
            st.markdown(f'<div class="tip-item tip-treatment"><span class="tip-dot-t">✓</span>{tip}</div>', unsafe_allow_html=True)

        # Prevention
        st.markdown('<div class="section-label">🛡️ Prevention</div>', unsafe_allow_html=True)
        for tip in info['prevention']:
            st.markdown(f'<div class="tip-item tip-prevention"><span class="tip-dot-p">→</span>{tip}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built by <strong>Kumari Pratibha Mani</strong> &nbsp;·&nbsp;
  <a href="https://github.com/pratibha2617">GitHub</a> &nbsp;·&nbsp;
  <a href="https://linkedin.com/in/kumari-pratibha-mani-973794294">LinkedIn</a> &nbsp;·&nbsp;
  Dataset: <a href="https://www.kaggle.com/datasets/emmarex/plantdisease">PlantVillage (Kaggle)</a>
  <br><br>🌿 Empowering Indian farmers with AI-powered crop disease detection
</div>
""", unsafe_allow_html=True)
