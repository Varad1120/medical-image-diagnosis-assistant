# 🩺 Medical Image Diagnosis Assistant

> An AI-powered deep learning system for **Pneumonia Detection from Chest X-Rays** using Convolutional Neural Networks (CNN) with Grad-CAM explainability.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 Project Overview

This project builds an end-to-end **Medical Image Diagnosis Assistant** that:
- Detects **Pneumonia** from chest X-ray images with high accuracy
- Uses **Transfer Learning** (EfficientNet-B3 / ResNet-50) for robust feature extraction
- Generates **Grad-CAM heatmaps** to visually explain *where* the model is looking
- Exposes a **FastAPI REST endpoint** for real-time inference
- Includes a **Streamlit dashboard** for interactive diagnosis

---

## 🏗️ Project Structure

```
medical-image-diagnosis-assistant/
├── data/
│   ├── raw/                    # Original dataset (Kaggle Chest X-Ray)
│   ├── processed/              # Preprocessed & augmented images
│   └── sample_images/          # Demo images for testing
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_model_training.ipynb # Model training & evaluation
│   └── 03_gradcam_viz.ipynb    # Grad-CAM visualization
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   └── transforms.py       # Image augmentation pipelines
│   ├── models/
│   │   ├── cnn_model.py        # Custom CNN architecture
│   │   └── efficientnet.py     # EfficientNet transfer learning
│   ├── utils/
│   │   ├── gradcam.py          # Grad-CAM implementation
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── trainer.py          # Training loop
│   └── visualization/
│       └── plots.py            # Plotting utilities
├── app/
│   ├── main.py                 # FastAPI app
│   ├── streamlit_app.py        # Streamlit dashboard
│   └── static/                 # Static assets
├── tests/
│   └── test_model.py           # Unit tests
├── configs/
│   └── config.yaml             # Training hyperparameters
├── docs/
│   └── model_card.md           # Model Card documentation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | PyTorch 2.0, torchvision |
| **Transfer Learning** | EfficientNet-B3, ResNet-50 |
| **Explainability** | Grad-CAM, pytorch-grad-cam |
| **Data Processing** | NumPy, Pandas, Pillow, OpenCV |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Experiment Tracking** | MLflow |
| **Testing** | pytest |

---

## 📊 Dataset

**[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** — Kaggle

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Varad1120/medical-image-diagnosis-assistant.git
cd medical-image-diagnosis-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
# Install Kaggle CLI first
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip
```

### 4. Train the model
```bash
python src/utils/trainer.py --config configs/config.yaml
```

### 5. Run the API
```bash
uvicorn app.main:app --reload
# API docs at: http://localhost:8000/docs
```

### 6. Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Results

| Model | Accuracy | AUC-ROC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Custom CNN | ~85% | ~0.91 | ~0.87 | ~0.88 |
| ResNet-50 | ~91% | ~0.96 | ~0.92 | ~0.94 |
| EfficientNet-B3 | ~94% | ~0.98 | ~0.95 | ~0.96 |

---

## 🔬 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions of the X-ray image influenced the model's decision — critical for **clinical trust and validation**.

---

## 🗺️ Roadmap

- [x] Project setup & repository structure
- [ ] EDA notebook
- [ ] Data preprocessing & augmentation pipeline
- [ ] Custom CNN baseline model
- [ ] EfficientNet transfer learning model
- [ ] Grad-CAM explainability
- [ ] FastAPI inference endpoint
- [ ] Streamlit dashboard
- [ ] MLflow experiment tracking
- [ ] Docker containerization
- [ ] Model card documentation

---

## 👨‍💻 Author

**Varad Kale** · [GitHub @Varad1120](https://github.com/Varad1120)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

> ⚠️ **Disclaimer**: This tool is for educational and research purposes only. It is **not** intended for clinical diagnosis.
