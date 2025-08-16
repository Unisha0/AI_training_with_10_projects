# app.py

import streamlit as st
import torch
from gan_model import Generator, Discriminator, DCGenerator, DCDiscriminator
from train import train_model


with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
st.set_page_config(page_title="GAN Image Generator", layout="centered")
st.title("🧠 GAN / DCGAN Mini Image Generator")

model_type = st.selectbox("Select Model Type", ["GAN", "DCGAN"])
epochs = st.slider("Number of Epochs", 1, 50, 10)
z_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if st.button("🚀 Train Now"):
    with st.spinner("Training in progress..."):
        if model_type == "GAN":
            gen = Generator(z_dim).to(device)
            disc = Discriminator().to(device)
        else:
            gen = DCGenerator(z_dim).to(device)
            disc = DCDiscriminator().to(device)

        train_model(model_type, gen, disc, z_dim, device, epochs)

    st.success("🎉 Training Complete!")
    st.image(f"generated/{model_type.lower()}_output_epoch{epochs}.png", use_column_width=True, caption="Generated Digits")

    