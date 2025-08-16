import streamlit as st
from inference import generate_pet
import os

st.set_page_config(page_title="PetMorphAI", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🐾 PetMorphAI - Generate Your Favorite Pets")

pet_types = ["Cat", "Dog", "Bird", "Rabbit", "Fish"]
pet_choice = st.selectbox("Choose a pet type:", pet_types)

if st.button("Generate Pet Image"):
    with st.spinner("Generating..."):
        label = pet_choice.lower()
        image_path = generate_pet(label, [p.lower() for p in pet_types], "models/generator_epoch_50.pt")
        st.image(image_path, caption=f"Generated {pet_choice}", use_column_width=True)
        st.success("Here is your pet!")