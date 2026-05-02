# 🎨 GAN & DCGAN — Mini Image AI Generator

> Training Generative Adversarial Networks (GAN) and Deep Convolutional GANs (DCGAN) to generate realistic synthetic images from random noise.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-GAN%20%7C%20DCGAN-purple) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project implements both a **vanilla GAN** and a **Deep Convolutional GAN (DCGAN)** to generate synthetic images. The models learn to produce realistic images by training a Generator network to fool a Discriminator network in an adversarial process. A Streamlit app allows users to generate new images interactively.

---

## ✨ Features

- 🧠 Vanilla GAN and DCGAN implementations
- 🏋️ Adversarial training pipeline (Generator vs Discriminator)
- 🖼️ Image generation from random latent vectors
- 📊 Training loss visualization
- 🌐 Streamlit app for interactive image generation

---

## 🗂️ Project Structure

```
4.TRAIN_GAN_AND_DCGAN(MINI _IMAGE_AI)/
├── app.py          # Streamlit interactive demo
├── gan_model.py    # GAN & DCGAN architecture
├── train.py        # Training loop
├── utils.py        # Utility functions
├── assets/         # Generated sample images
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd "AI_training_with_10_projects/4.TRAIN_GAN_AND_DCGAN(MINI _IMAGE_AI)"

pip install torch torchvision streamlit matplotlib numpy
```

### Train the GAN

```bash
python train.py
```

### Launch Demo App

```bash
streamlit run app.py
```

---

## 🧠 How GANs Work

```
Random Noise → Generator → Fake Image
                                ↓
             Real Image → Discriminator → Real / Fake?
                                ↓
              Backpropagation to improve both networks
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core logic |
| PyTorch | GAN model training |
| Streamlit | Interactive image generation |
| Matplotlib | Training visualization |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

---

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
