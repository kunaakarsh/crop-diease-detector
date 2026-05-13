"""
Crop Disease Detection - ML Training Script
Uses color histogram + texture (LBP + Haralick) features with Random Forest
Simulates PlantVillage dataset class structure with synthetic training data
when real data is not available. In production, replace generate_synthetic_data()
with actual image loading from the PlantVillage Kaggle dataset.
"""

import numpy as np
import joblib
import os
from PIL import Image
import io

# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_color_histogram(img_array, bins=32):
    """Extract color histogram features from RGB image."""
    features = []
    for channel in range(3):
        hist, _ = np.histogram(img_array[:, :, channel], bins=bins, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-8)
        features.extend(hist)
    return np.array(features)

def extract_texture_features(gray):
    """Extract simple texture features using GLCM-like statistics."""
    features = []
    # Mean and variance
    features.append(gray.mean() / 255.0)
    features.append(gray.std() / 255.0)
    # Gradient magnitude (edge density)
    gy = np.diff(gray.astype(float), axis=0)
    gx = np.diff(gray.astype(float), axis=1)
    grad = np.abs(gy[:, :-1]) + np.abs(gx[:-1, :])
    features.append(grad.mean() / 255.0)
    features.append(grad.std() / 255.0)
    # Block-level variance (coarseness)
    h, w = gray.shape
    block_vars = []
    bh, bw = h // 4, w // 4
    for r in range(4):
        for c in range(4):
            block = gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            block_vars.append(block.var())
    features.extend(np.array(block_vars) / (255.0**2))
    return np.array(features)

def extract_green_ratio(img_array):
    """Extract green channel dominance features."""
    r, g, b = img_array[:,:,0].astype(float), img_array[:,:,1].astype(float), img_array[:,:,2].astype(float)
    total = r + g + b + 1e-8
    green_ratio = (g / total).mean()
    excess_green = (2*g - r - b).mean() / 255.0
    # Brown/yellow detection (disease indicators)
    brown = ((r > 100) & (g < 150) & (b < 80)).mean()
    yellow = ((r > 150) & (g > 150) & (b < 80)).mean()
    dark_spots = ((r < 80) & (g < 80) & (b < 80)).mean()
    return np.array([green_ratio, excess_green, brown, yellow, dark_spots])

def extract_features(img_array):
    """Full feature extraction pipeline."""
    img_resized = np.array(Image.fromarray(img_array).resize((128, 128)))
    gray = (0.299*img_resized[:,:,0] + 0.587*img_resized[:,:,1] + 0.114*img_resized[:,:,2]).astype(np.uint8)
    color_feats = extract_color_histogram(img_resized, bins=32)  # 96
    texture_feats = extract_texture_features(gray)               # 20
    green_feats = extract_green_ratio(img_resized)               # 5
    return np.concatenate([color_feats, texture_feats, green_feats])

# ── Synthetic data generation (replaces real dataset when unavailable) ─────────

DISEASE_CLASSES = [
    # (class_name, display_name, crop, severity, color_profile)
    ("tomato_healthy",           "Healthy Tomato",              "Tomato",   "None",     "healthy"),
    ("tomato_early_blight",      "Tomato Early Blight",         "Tomato",   "Moderate", "brown_spots"),
    ("tomato_late_blight",       "Tomato Late Blight",          "Tomato",   "Severe",   "dark_lesions"),
    ("tomato_leaf_mold",         "Tomato Leaf Mold",            "Tomato",   "Moderate", "yellow_green"),
    ("tomato_septoria_leaf_spot","Tomato Septoria Leaf Spot",   "Tomato",   "Moderate", "small_spots"),
    ("potato_healthy",           "Healthy Potato",              "Potato",   "None",     "healthy"),
    ("potato_early_blight",      "Potato Early Blight",         "Potato",   "Moderate", "brown_spots"),
    ("potato_late_blight",       "Potato Late Blight",          "Potato",   "Severe",   "dark_lesions"),
    ("corn_healthy",             "Healthy Corn",                "Corn",     "None",     "healthy"),
    ("corn_gray_leaf_spot",      "Corn Gray Leaf Spot",         "Corn",     "Moderate", "gray_lesions"),
    ("corn_northern_blight",     "Corn Northern Leaf Blight",   "Corn",     "Severe",   "tan_lesions"),
    ("corn_rust",                "Corn Common Rust",            "Corn",     "Moderate", "rust_spots"),
    ("grape_healthy",            "Healthy Grape",               "Grape",    "None",     "healthy"),
    ("grape_black_rot",          "Grape Black Rot",             "Grape",    "Severe",   "dark_lesions"),
    ("grape_esca",               "Grape Esca (Black Measles)",  "Grape",    "Severe",   "brown_spots"),
    ("apple_healthy",            "Healthy Apple",               "Apple",    "None",     "healthy"),
    ("apple_scab",               "Apple Scab",                  "Apple",    "Moderate", "dark_spots"),
    ("apple_black_rot",          "Apple Black Rot",             "Apple",    "Severe",   "dark_lesions"),
    ("apple_cedar_rust",         "Apple Cedar Apple Rust",      "Apple",    "Moderate", "rust_spots"),
    ("pepper_healthy",           "Healthy Pepper",              "Pepper",   "None",     "healthy"),
    ("pepper_bacterial_spot",    "Pepper Bacterial Spot",       "Pepper",   "Moderate", "small_spots"),
    ("rice_healthy",             "Healthy Rice",                "Rice",     "None",     "healthy"),
    ("rice_blast",               "Rice Blast",                  "Rice",     "Severe",   "gray_lesions"),
    ("rice_brown_spot",          "Rice Brown Spot",             "Rice",     "Moderate", "brown_spots"),
    ("wheat_healthy",            "Healthy Wheat",               "Wheat",    "None",     "healthy"),
    ("wheat_rust",               "Wheat Leaf Rust",             "Wheat",    "Severe",   "rust_spots"),
]

def generate_synthetic_image(profile, seed):
    """Generate a synthetic plant leaf image with disease-specific color/texture."""
    rng = np.random.RandomState(seed)
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    profiles = {
        "healthy":      {"base": [45, 140, 35],  "noise": 30, "spots": False},
        "brown_spots":  {"base": [60, 120, 40],  "noise": 35, "spots": True,  "spot_color": [120, 70, 30]},
        "dark_lesions": {"base": [50, 100, 35],  "noise": 40, "spots": True,  "spot_color": [40,  30, 20]},
        "yellow_green": {"base": [90, 150, 50],  "noise": 30, "spots": True,  "spot_color": [160, 150, 40]},
        "small_spots":  {"base": [55, 130, 40],  "noise": 25, "spots": True,  "spot_color": [90,  60, 20]},
        "gray_lesions": {"base": [65, 125, 45],  "noise": 35, "spots": True,  "spot_color": [120, 120, 110]},
        "tan_lesions":  {"base": [60, 115, 40],  "noise": 30, "spots": True,  "spot_color": [140, 110, 70]},
        "rust_spots":   {"base": [50, 120, 35],  "noise": 30, "spots": True,  "spot_color": [160, 80,  20]},
        "dark_spots":   {"base": [55, 125, 40],  "noise": 28, "spots": True,  "spot_color": [50,  40, 30]},
    }
    p = profiles[profile]
    base = np.array(p["base"])
    noise = rng.randint(-p["noise"], p["noise"], (128, 128, 3))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)

    if p.get("spots"):
        n_spots = rng.randint(5, 25)
        for _ in range(n_spots):
            cx = rng.randint(10, 118)
            cy = rng.randint(10, 118)
            r  = rng.randint(3, 14)
            yy, xx = np.ogrid[:128, :128]
            mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
            spot_noise = rng.randint(-15, 15, 3)
            sc = np.clip(np.array(p["spot_color"]) + spot_noise, 0, 255)
            img[mask] = sc

    return img

def generate_training_data(n_per_class=200):
    """Generate synthetic training dataset."""
    X, y = [], []
    for idx, (class_key, _, _, _, profile) in enumerate(DISEASE_CLASSES):
        for i in range(n_per_class):
            img = generate_synthetic_image(profile, seed=idx * 10000 + i)
            feats = extract_features(img)
            X.append(feats)
            y.append(idx)
    return np.array(X), np.array(y)

# ── Training ───────────────────────────────────────────────────────────────────

def train():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.svm import SVC

    print("Generating training data...")
    X, y = generate_training_data(n_per_class=300)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(DISEASE_CLASSES)} classes")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Random Forest classifier...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    model_path = os.path.join(os.path.dirname(__file__), "models", "crop_disease_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    return pipeline

if __name__ == "__main__":
    train()
