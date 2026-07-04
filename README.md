# 🌿 CropGuard AI — Crop Disease Detector

An AI-powered web application that detects crop diseases from leaf images and provides treatment recommendations — built to help Indian farmers identify and respond to crop diseases early.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.4-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Problem Statement

Indian farmers lose 20–30% of their crops annually to diseases that go undetected until it's too late. Most farmers in Tier 2/3 cities and rural areas have no access to agronomists. This app gives every farmer an AI agronomist in their pocket.

---

## ✨ Features

- 📷 Upload any leaf photo (JPG/PNG)
- 🦠 Instant disease detection with confidence score
- 💊 Detailed treatment recommendations
- 🛡️ Prevention tips for each disease
- 🌐 Works on mobile browsers
- ⚡ Results in under 2 seconds

---

## 🧠 How It Works

1. **Image Preprocessing** — OpenCV resizes and normalizes leaf images
2. **Feature Extraction** — Extracts HSV color histograms + texture gradient features
3. **ML Classification** — Random Forest classifier trained on 50,000+ PlantVillage images
4. **Result Display** — Disease name, confidence score, treatment, and prevention advice

---

## 🌱 Supported Crops & Diseases

| Crop    | Diseases Detected                          |
|---------|--------------------------------------------|
| Tomato  | Late Blight, Early Blight, Healthy         |
| Potato  | Late Blight, Early Blight, Healthy         |
| Pepper  | Bacterial Spot, Healthy                    |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/pratibha2617/crop-disease-detector.git
cd crop-disease-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to: https://www.kaggle.com/datasets/emmarex/plantdisease
- Download and extract `PlantVillage` folder into the project root

### 4. Train the model
```bash
python train_model.py
```
Expected output: **~89% accuracy** on test set

### 5. Run the app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~89%   |
| Precision | ~88%   |
| Recall    | ~87%   |
| F1 Score  | ~87%   |

*Trained on 50,000+ images from the PlantVillage dataset*

---

## 🛠️ Tech Stack

| Component       | Technology                    |
|-----------------|-------------------------------|
| Frontend/UI     | Streamlit                     |
| ML Model        | Scikit-Learn (Random Forest)  |
| Image Processing| OpenCV                        |
| Feature Engineering | HSV Histograms + Gradients |
| Data Handling   | NumPy, Pandas                 |
| Deployment      | Streamlit Cloud (free)        |

---

## 📁 Project Structure

```
crop_disease_detector/
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── model/
│   ├── crop_disease_model.pkl   # Trained model (after training)
│   └── label_encoder.pkl        # Label encoder (after training)
├── utils/
│   ├── feature_extractor.py     # OpenCV feature extraction
│   └── disease_info.py          # Disease treatment database
└── README.md
```

---

## 🌐 Deploy for Free

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Click Deploy — live in 2 minutes!

---

## 👩‍💻 Author

**Kumari Pratibha Mani**
- GitHub: [@pratibha2617](https://github.com/pratibha2617)
- LinkedIn: [kumari-pratibha-mani](https://linkedin.com/in/kumari-pratibha-mani-973794294)
- Email: pratibha2617@gmail.com

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Dataset credit: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — Hughes & Salathé (2015)*
