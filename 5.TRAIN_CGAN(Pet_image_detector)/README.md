# 🐾 Conditional GAN — Pet Image Detector

> A Conditional Generative Adversarial Network (CGAN) trained to generate and detect pet images conditioned on class labels.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CGAN-purple) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project trains a **Conditional GAN (CGAN)** on pet image data. Unlike standard GANs, a CGAN conditions both the Generator and Discriminator on class labels (e.g., cat, dog), enabling controlled image generation. An inference pipeline and Streamlit app are included for demo purposes.

---

## ✨ Features

- 🐱🐶 Class-conditioned image generation (cats, dogs, etc.)
- 🏋️ Full CGAN training pipeline
- 🔮 Inference script for generating new pet images
- 📁 Pre-organized model saving structure
- 🌐 Streamlit demo app

---

## 🗂️ Project Structure

```
5.TRAIN_CGAN(Pet_image_detector)/
├── app.py           # Streamlit demo application
├── train.py         # CGAN training loop
├── inference.py     # Generate images from trained model
├── utils.py         # Helper functions
├── models/          # Saved model checkpoints
├── assets/          # Sample generated images
├── requirements.txt # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd "AI_training_with_10_projects/5.TRAIN_CGAN(Pet_image_detector)"

pip install -r requirements.txt
```

### Train the CGAN

```bash
python train.py
```

### Run Inference

```bash
python inference.py
```

### Launch Demo

```bash
streamlit run app.py
```

---

## 🧠 CGAN Architecture

```
Label (one-hot) + Noise → Generator → Conditioned Fake Image
Label + Real/Fake Image → Discriminator → Real / Fake?
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core logic |
| PyTorch | CGAN model |
| Streamlit | Demo interface |
| Oxford-IIIT Pet Dataset | Training data |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

---

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
