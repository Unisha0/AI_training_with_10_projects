# 🖼️ CNN Image Classification — MNIST Digit Recognizer

> A Convolutional Neural Network (CNN) built from scratch to classify handwritten digits from the MNIST dataset, served via an interactive Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-purple) ![Dataset](https://img.shields.io/badge/Dataset-MNIST-yellow) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) from the classic **MNIST dataset**. The model is trained using PyTorch/TensorFlow, and the predictions are served through a Streamlit web interface where users can draw or upload digits for classification.

---

## ✨ Features

- 🧠 Custom CNN architecture for image classification
- 🏋️ Model training with accuracy/loss tracking
- 🌐 Streamlit web app for live digit prediction
- 🗃️ MNIST JPG dataset integration
- 🛠️ Modular codebase (model, training, utils separated)

---

## 🗂️ Project Structure

```
3.CNN_IMAGE/
├── app.py                  # Streamlit web app
├── cnn_model.py            # CNN architecture definition
├── train_model.py          # Model training script
├── utils.py                # Helper functions
├── MNIST - JPG - training/ # Training dataset
├── assets/                 # Images and assets
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd AI_training_with_10_projects/3.CNN_IMAGE

pip install torch torchvision streamlit pillow numpy
```

### Train the Model

```bash
python train_model.py
```

### Launch the App

```bash
streamlit run app.py
```

---

## 🧪 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | ~99% |
| Epochs | 10 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core logic |
| PyTorch / TensorFlow | CNN model |
| Streamlit | Web interface |
| MNIST Dataset | Training data |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

---

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
