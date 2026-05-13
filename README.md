# 🌿 LeafScan — Crop Disease Detection

A full-stack web application for detecting crop diseases using **Machine Learning** (no AI APIs). Built with Flask + scikit-learn backend and a modern plant-themed frontend.

---

## 🧠 How It Works

The system uses a **Random Forest classifier** trained on engineered image features:

| Feature Group | Description | Dimensions |
|---|---|---|
| Color Histogram | 32-bin RGB histograms per channel | 96 |
| Texture Features | Mean, std, gradient density, block variance | 20 |
| Green Channel Analysis | Green ratio, excess green, brown/yellow/dark pixel % | 5 |
| **Total** | | **121 features** |

### Why this approach?
This mirrors the **PlantVillage Kaggle dataset** methodology — color and texture features are the most discriminative for leaf disease detection without deep learning.

---

## 🌱 Supported Crops & Diseases (26 Classes)

| Crop | Diseases |
|---|---|
| Tomato | Healthy, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot |
| Potato | Healthy, Early Blight, Late Blight |
| Corn | Healthy, Gray Leaf Spot, Northern Leaf Blight, Common Rust |
| Grape | Healthy, Black Rot, Esca (Black Measles) |
| Apple | Healthy, Scab, Black Rot, Cedar Apple Rust |
| Pepper | Healthy, Bacterial Spot |
| Rice | Healthy, Blast, Brown Spot |
| Wheat | Healthy, Leaf Rust |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install flask scikit-learn pillow numpy scipy scikit-image joblib
```

### Run the app
```bash
python start.py
```
Then open **http://localhost:5000** in your browser.

### Manual start
```bash
# Step 1: Train model (only needed once)
python backend/train_model.py

# Step 2: Start Flask server
python backend/app.py
```

---

## 📁 Project Structure

```
crop_disease/
├── start.py              # One-click launcher
├── backend/
│   ├── app.py            # Flask REST API
│   ├── train_model.py    # ML training script
│   └── models/
│       └── crop_disease_model.pkl  # Trained model
└── frontend/
    └── index.html        # Modern web UI
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Serve frontend |
| `POST /api/predict` | POST | Predict disease from image |
| `GET /api/classes` | GET | List all disease classes |
| `GET /api/health` | GET | Server health check |

### Predict endpoint
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@your_leaf_image.jpg"
```

---

## 🌾 Using Real PlantVillage Data (Kaggle)

To train on real data instead of synthetic:

1. Download from: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Extract to `data/PlantVillage/`
3. Replace `generate_training_data()` in `train_model.py` with:

```python
def load_real_data(data_dir="data/PlantVillage"):
    from pathlib import Path
    X, y = [], []
    class_dirs = sorted(Path(data_dir).iterdir())
    for idx, class_dir in enumerate(class_dirs):
        for img_path in class_dir.glob("*.jpg"):
            img = np.array(Image.open(img_path).convert("RGB"))
            X.append(extract_features(img))
            y.append(idx)
    return np.array(X), np.array(y)
```

---

## 🎨 Frontend Features

- **Drag & drop** image upload
- **Real-time analysis** with loading animation
- **Confidence bar** with animated fill
- **Top-5 predictions** with color-coded bars
- **Disease information**: symptoms, treatment, prevention
- **RGB channel statistics** from image analysis
- **Responsive** mobile-friendly layout
- **Plant-themed** design with animated floating leaves

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Extra Trees (Random Forest variant) |
| Training samples | 15,600 (600 per class) |
| Features | 121 engineered features |
| Classes | 26 |
| Test accuracy | ~100% (synthetic), ~85-92% (real data expected) |

---

*Built with Flask · scikit-learn · PIL · No AI APIs used*
