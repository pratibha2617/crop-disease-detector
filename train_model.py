"""
train_model.py — Train the crop disease detection model.

SETUP INSTRUCTIONS:
1. Download PlantVillage dataset from Kaggle:
   https://www.kaggle.com/datasets/emmarex/plantdisease
2. Extract it so your folder looks like:
   PlantVillage/
     Tomato___Late_blight/  (folder with .jpg images)
     Tomato___Early_blight/
     Tomato___healthy/
     Potato___Late_blight/
     ... etc
3. Set DATASET_PATH below to point to your PlantVillage folder
4. Run: python train_model.py
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import cv2
from utils.feature_extractor import extract_features

# ── CONFIG ─────────────────────────────────────────────────────
DATASET_PATH = "PlantVillage"   # Change this to your dataset folder path
MODEL_SAVE_PATH = "model/crop_disease_model.pkl"
LABEL_SAVE_PATH = "model/label_encoder.pkl"
MAX_IMAGES_PER_CLASS = 300      # Reduce if RAM is limited
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# ── LOAD DATA ──────────────────────────────────────────────────
def load_dataset(dataset_path):
    print(f"Loading dataset from: {dataset_path}")
    X, y = [], []
    classes = sorted(os.listdir(dataset_path))

    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir)
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        images = images[:MAX_IMAGES_PER_CLASS]

        print(f"  Loading {len(images):>4} images from: {class_name}")

        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            features = extract_features(img)
            X.append(features)
            y.append(class_name)

    print(f"\nTotal samples loaded: {len(X)}")
    return np.array(X), np.array(y)

# ── MAIN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load dataset
    X, y = load_dataset(DATASET_PATH)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes found: {list(le.classes_)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nTraining samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Train Random Forest
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(le, LABEL_SAVE_PATH)
    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✅ Label encoder saved to: {LABEL_SAVE_PATH}")
    print("\nNow run: streamlit run app.py")
