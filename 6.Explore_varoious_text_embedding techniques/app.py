# app.py
import streamlit as st
from utils import get_tfidf_embedding, get_w2v_embedding, load_models
import numpy as np

st.set_page_config(page_title="Text Embedding Explorer", layout="centered")
st.title("🔤 Text Embedding with TF-IDF & Word2Vec")

# Load models (TF-IDF + Word2Vec)
tfidf_vectorizer, w2v_model = load_models()

# Input text
user_input = st.text_area("Enter a sentence:", "Natural language processing is fascinating.")

# Embedding method
method = st.selectbox("Choose embedding method:", ["TF-IDF", "Word2Vec"])

if st.button("Generate Embedding"):
    if method == "TF-IDF":
        embedding = get_tfidf_embedding(user_input, tfidf_vectorizer)
    else:
        embedding = get_w2v_embedding(user_input, w2v_model)

    st.subheader(f"{method} Embedding Vector:")
    st.write(embedding)
    st.success(f"Embedding vector shape: {embedding.shape}")