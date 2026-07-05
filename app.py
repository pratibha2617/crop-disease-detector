import streamlit as st
import numpy as np
import cv2
import joblib
import os
import requests
from PIL import Image
from utils.feature_extractor import extract_features
from utils.disease_info import get_disease_info
 
st.set_page_config(
    page_title="CropGuard AI — Crop Disease Detector",
    page_icon="🌿",
    layout="wide"
)
 
# ── Language Data ───────────────────────────────────────────────
LANG = {
    "en": {
        "title": "CropGuard AI",
        "subtitle": "AI-powered crop disease detection for Indian farmers — instant diagnosis, expert treatment advice",
        "upload_title": "📷 Upload Leaf Image",
        "upload_hint": "📌 Use a clear, well-lit photo of a single leaf for best accuracy",
        "result_title": "🔬 Analysis Result",
        "empty_title": "No image uploaded yet",
        "empty_sub": "Upload a leaf photo on the left to get instant AI diagnosis",
        "analyzing": "🔬 Analyzing your leaf...",
        "confidence": "Confidence",
        "cause": "🧬 Cause",
        "symptoms": "🔍 Symptoms",
        "treatment": "💊 Treatment",
        "prevention": "🛡️ Prevention",
        "weather_title": "🌦️ Disease Risk Forecast",
        "weather_high": "🔴 HIGH Disease Risk Today",
        "weather_medium": "🟡 MODERATE Disease Risk Today",
        "weather_low": "🟢 LOW Disease Risk Today",
        "weather_high_msg": "High humidity + warm temperature = ideal conditions for fungal diseases. Inspect crops carefully today!",
        "weather_medium_msg": "Moderate conditions — monitor your crops closely over next 48 hours.",
        "weather_low_msg": "Conditions are not favorable for most fungal diseases today.",
        "humidity": "Humidity",
        "temperature": "Temperature",
        "weather_error": "Weather data unavailable",
        "language": "🌐 Language / भाषा",
        "accuracy": "ACCURACY",
        "diseases": "DISEASES",
        "images": "IMAGES TRAINED",
        "speed": "DETECTION SPEED",
        "footer": "Helping Indian farmers detect crop diseases with AI",
        "healthy_msg": "Plant is Healthy!",
        "disease_msg": "Disease Detected!"
    },
    "hi": {
        "title": "क्रॉपगार्ड AI",
        "subtitle": "भारतीय किसानों के लिए AI-आधारित फसल रोग पहचान — तुरंत निदान, विशेषज्ञ उपचार सलाह",
        "upload_title": "📷 पत्ती की फोटो अपलोड करें",
        "upload_hint": "📌 सबसे अच्छे परिणाम के लिए एक साफ, अच्छी रोशनी वाली पत्ती की फोटो का उपयोग करें",
        "result_title": "🔬 विश्लेषण परिणाम",
        "empty_title": "अभी तक कोई फोटो अपलोड नहीं हुई",
        "empty_sub": "तुरंत AI निदान पाने के लिए बाईं ओर पत्ती की फोटो अपलोड करें",
        "analyzing": "🔬 आपकी पत्ती का विश्लेषण हो रहा है...",
        "confidence": "विश्वास स्तर",
        "cause": "🧬 कारण",
        "symptoms": "🔍 लक्षण",
        "treatment": "💊 उपचार",
        "prevention": "🛡️ बचाव",
        "weather_title": "🌦️ रोग जोखिम पूर्वानुमान",
        "weather_high": "🔴 आज रोग का खतरा अधिक है",
        "weather_medium": "🟡 आज रोग का खतरा मध्यम है",
        "weather_low": "🟢 आज रोग का खतरा कम है",
        "weather_high_msg": "अधिक नमी + गर्म तापमान = फफूंद रोगों के लिए आदर्श स्थिति। आज फसल की सावधानी से जांच करें!",
        "weather_medium_msg": "मध्यम स्थितियां — अगले 48 घंटों में अपनी फसल पर ध्यान दें।",
        "weather_low_msg": "आज अधिकांश फफूंद रोगों के लिए परिस्थितियां अनुकूल नहीं हैं।",
        "humidity": "नमी",
        "temperature": "तापमान",
        "weather_error": "मौसम डेटा उपलब्ध नहीं है",
        "language": "🌐 Language / भाषा",
        "accuracy": "सटीकता",
        "diseases": "रोग",
        "images": "प्रशिक्षण फोटो",
        "speed": "पहचान गति",
        "footer": "AI से भारतीय किसानों की फसल रोग पहचान में मदद",
        "healthy_msg": "पौधा स्वस्थ है!",
        "disease_msg": "रोग पाया गया!"
    }
}
 
# ── Weather Risk ────────────────────────────────────────────────
def get_weather_risk(city="New Delhi"):
    try:
        API_KEY = "bd5e378503939ddaee76f12ad7a97608"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        r = requests.get(url, timeout=5)
        data = r.json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        risk = "high" if humidity > 80 and 15 < temp < 35 else "medium" if humidity > 60 else "low"
        return {"temp": temp, "humidity": humidity, "risk": risk, "city": city}
    except:
        return None
 
# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stApp { background: #f0fdf4; }
.hero {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 40%, #166534 100%);
    border-radius: 24px; padding: 50px 40px 40px; text-align: center;
    margin-bottom: 28px; box-shadow: 0 8px 32px rgba(22,163,74,0.25);
}
.hero-title { font-size: 2.8em; font-weight: 800; color: white; margin: 0 0 8px 0; letter-spacing: -1px; }
.hero-sub { color: #bbf7d0; font-size: 1em; font-weight: 400; margin-bottom: 30px; }
.stats-row { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
.stat-pill {
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 50px; padding: 10px 24px; color: white; backdrop-filter: blur(10px);
}
.stat-num { font-size: 1.4em; font-weight: 700; display: block; }
.stat-lbl { font-size: 0.7em; text-transform: uppercase; letter-spacing: 1.5px; color: #bbf7d0; }
.crop-row { display: flex; justify-content: center; gap: 10px; margin-bottom: 24px; flex-wrap: wrap; }
.crop-tag {
    background: white; border: 1.5px solid #86efac; border-radius: 50px;
    padding: 6px 18px; font-size: 0.85em; font-weight: 500; color: #15803d;
    box-shadow: 0 2px 8px rgba(22,163,74,0.1);
}
.panel {
    background: white; border-radius: 20px; padding: 28px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 1px solid #dcfce7;
}
.panel-title { font-size: 1em; font-weight: 600; color: #15803d; margin-bottom: 16px; }
.upload-hint {
    background: #f0fdf4; border: 2px dashed #86efac; border-radius: 14px;
    padding: 20px; text-align: center; color: #16a34a; font-size: 0.9em; margin-bottom: 16px;
}
.empty-state { text-align: center; padding: 60px 20px; color: #6b7280; }
.empty-icon { font-size: 3.5em; margin-bottom: 12px; }
.empty-title { font-size: 1em; font-weight: 600; color: #374151; margin-bottom: 6px; }
.empty-sub { font-size: 0.85em; color: #9ca3af; }
.result-disease {
    background: linear-gradient(135deg, #fef2f2, #fff);
    border: 1.5px solid #fca5a5; border-radius: 16px; padding: 20px; margin-bottom: 16px;
}
.result-healthy {
    background: linear-gradient(135deg, #f0fdf4, #fff);
    border: 1.5px solid #86efac; border-radius: 16px; padding: 20px; margin-bottom: 16px;
}
.result-icon { font-size: 2.2em; margin-bottom: 8px; }
.result-name { font-size: 1.25em; font-weight: 700; color: #111827; margin-bottom: 4px; }
.result-conf { font-size: 0.85em; color: #6b7280; font-weight: 500; }
.conf-wrap { background: #f3f4f6; border-radius: 999px; height: 8px; margin: 12px 0 20px; overflow: hidden; }
.info-card { border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }
.info-card-cause { background: #fff1f2; border-left: 4px solid #f43f5e; }
.info-card-symptom { background: #fffbeb; border-left: 4px solid #f59e0b; }
.info-card-label { font-size: 0.72em; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; color: #6b7280; }
.info-card-text { font-size: 0.9em; color: #374151; line-height: 1.55; }
.tip-item { display: flex; gap: 10px; align-items: flex-start; padding: 10px 14px; border-radius: 10px; margin-bottom: 7px; font-size: 0.88em; color: #374151; line-height: 1.5; }
.tip-treatment { background: #f0fdf4; border: 1px solid #bbf7d0; }
.tip-prevention { background: #fefce8; border: 1px solid #fde68a; }
.tip-dot-t { color: #16a34a; font-size: 1.1em; flex-shrink: 0; margin-top: 1px; }
.tip-dot-p { color: #d97706; font-size: 1.1em; flex-shrink: 0; margin-top: 1px; }
.section-label { font-size: 0.85em; font-weight: 700; color: #374151; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 0.5px; }
.weather-high { background: linear-gradient(135deg, #fef2f2, #fff); border: 1.5px solid #fca5a5; border-radius: 16px; padding: 16px 20px; margin-bottom: 20px; }
.weather-medium { background: linear-gradient(135deg, #fffbeb, #fff); border: 1.5px solid #fde68a; border-radius: 16px; padding: 16px 20px; margin-bottom: 20px; }
.weather-low { background: linear-gradient(135deg, #f0fdf4, #fff); border: 1.5px solid #86efac; border-radius: 16px; padding: 16px 20px; margin-bottom: 20px; }
.weather-title { font-size: 0.9em; font-weight: 700; color: #374151; margin-bottom: 6px; }
.weather-msg { font-size: 0.82em; color: #6b7280; line-height: 1.5; }
.weather-stats { display: flex; gap: 16px; margin-top: 10px; }
.weather-stat { font-size: 0.82em; color: #374151; font-weight: 500; }
.lang-toggle { display: flex; justify-content: flex-end; margin-bottom: 16px; }
.footer { background: white; border-radius: 14px; padding: 18px 24px; text-align: center; margin-top: 24px; border: 1px solid #dcfce7; font-size: 0.85em; color: #6b7280; }
.footer a { color: #16a34a; text-decoration: none; font-weight: 500; }
</style>
""", unsafe_allow_html=True)
 
# ── Language Toggle ─────────────────────────────────────────────
col_lang = st.columns([4, 1])
with col_lang[1]:
    lang = st.selectbox("🌐", ["English", "हिंदी"], label_visibility="collapsed")
L = LANG["hi"] if lang == "हिंदी" else LANG["en"]
 
# ── Load Model ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("model/crop_disease_model.pkl") and os.path.exists("model/label_encoder.pkl"):
        return joblib.load("model/crop_disease_model.pkl"), joblib.load("model/label_encoder.pkl")
    return None, None
 
model, label_encoder = load_model()
 
# ── HERO ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-title">🌿 {L['title']}</div>
  <div class="hero-sub">{L['subtitle']}</div>
  <div class="stats-row">
    <div class="stat-pill"><span class="stat-num">91%</span><span class="stat-lbl">{L['accuracy']}</span></div>
    <div class="stat-pill"><span class="stat-num">15</span><span class="stat-lbl">{L['diseases']}</span></div>
    <div class="stat-pill"><span class="stat-num">50K+</span><span class="stat-lbl">{L['images']}</span></div>
    <div class="stat-pill"><span class="stat-num">&lt;2s</span><span class="stat-lbl">{L['speed']}</span></div>
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
 
# ── Weather Risk ────────────────────────────────────────────────
with st.expander(f"🌦️ {L['weather_title']}", expanded=True):
    city = st.text_input("Enter your city / अपना शहर दर्ज करें", value="New Delhi", label_visibility="collapsed")
    weather = get_weather_risk(city)
    if weather:
        risk = weather['risk']
        css_class = f"weather-{risk}"
        title = L[f'weather_{risk}']
        msg = L[f'weather_{risk}_msg']
        st.markdown(f"""
        <div class="{css_class}">
          <div class="weather-title">{title}</div>
          <div class="weather-msg">{msg}</div>
          <div class="weather-stats">
            <span class="weather-stat">💧 {L['humidity']}: {weather['humidity']}%</span>
            <span class="weather-stat">🌡️ {L['temperature']}: {weather['temp']}°C</span>
            <span class="weather-stat">📍 {weather['city']}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(L['weather_error'])
 
# ── Two Column Layout ───────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")
 
with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">{L["upload_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="upload-hint">{L["upload_hint"]}</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=380)
    st.markdown('</div>', unsafe_allow_html=True)
 
with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">{L["result_title"]}</div>', unsafe_allow_html=True)
 
    if not uploaded_file:
        st.markdown(f"""
        <div class="empty-state">
          <div class="empty-icon">🔬</div>
          <div class="empty-title">{L['empty_title']}</div>
          <div class="empty-sub">{L['empty_sub']}</div>
        </div>
        """, unsafe_allow_html=True)
    elif model is None:
        st.warning("⚠️ Model not found. Please run train_model.py first.")
    else:
        with st.spinner(L['analyzing']):
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
 
        if is_healthy:
            st.markdown(f"""
            <div class="result-healthy">
              <div class="result-icon">✅</div>
              <div class="result-name">{display_name}</div>
              <div class="result-conf">{L['confidence']}: {confidence}%</div>
            </div>
            """, unsafe_allow_html=True)
            fill_color = "#16a34a"
        else:
            st.markdown(f"""
            <div class="result-disease">
              <div class="result-icon">🦠</div>
              <div class="result-name">{display_name}</div>
              <div class="result-conf">{L['confidence']}: {confidence}%</div>
            </div>
            """, unsafe_allow_html=True)
            fill_color = "#16a34a" if confidence >= 70 else "#f59e0b" if confidence >= 40 else "#ef4444"
 
        st.markdown(f"""
        <div class="conf-wrap">
          <div style="background:{fill_color}; width:{confidence}%; height:8px; border-radius:999px;"></div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown(f"""
        <div class="info-card info-card-cause">
          <div class="info-card-label">{L['cause']}</div>
          <div class="info-card-text">{info['cause']}</div>
        </div>
        <div class="info-card info-card-symptom">
          <div class="info-card-label">{L['symptoms']}</div>
          <div class="info-card-text">{info['symptoms']}</div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown(f'<div class="section-label">{L["treatment"]}</div>', unsafe_allow_html=True)
        for tip in info['treatment']:
            st.markdown(f'<div class="tip-item tip-treatment"><span class="tip-dot-t">✓</span>{tip}</div>', unsafe_allow_html=True)
 
        st.markdown(f'<div class="section-label">{L["prevention"]}</div>', unsafe_allow_html=True)
        for tip in info['prevention']:
            st.markdown(f'<div class="tip-item tip-prevention"><span class="tip-dot-p">→</span>{tip}</div>', unsafe_allow_html=True)
 
    st.markdown('</div>', unsafe_allow_html=True)
 
# ── Footer ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  Built by <strong>Kumari Pratibha Mani</strong> &nbsp;·&nbsp;
  <a href="https://github.com/pratibha2617">GitHub</a> &nbsp;·&nbsp;
  <a href="https://linkedin.com/in/kumari-pratibha-mani-973794294">LinkedIn</a> &nbsp;·&nbsp;
  Dataset: <a href="https://www.kaggle.com/datasets/emmarex/plantdisease">PlantVillage</a>
  <br><br>🌿 {L['footer']}
</div>
""", unsafe_allow_html=True)
 
