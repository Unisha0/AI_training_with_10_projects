# utils.py

import numpy as np
import cv2
import torch
from torchvision import transforms
from cnn_model import CNN
import os
from PIL import Image
import streamlit as st

MODEL_PATH = "generated/model_cnn.pt"

# Load model once and reuse
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    """Convert PIL image to 28x28 tensor."""
    image = image.convert('L')  # grayscale
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_tensor = transforms.ToTensor()(img_array).unsqueeze(0)
    img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor)
    return img_tensor

def predict(image_tensor):
    model = load_model()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

def load_random_sample():
    from torchvision import datasets
    test_data = datasets.MNIST(root='assets/', train=False, download=True)
    idx = np.random.randint(0, len(test_data))
    img, label = test_data[idx]
    return img, label

def inject_custom_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)