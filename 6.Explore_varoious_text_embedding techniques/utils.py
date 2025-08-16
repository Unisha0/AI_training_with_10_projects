# utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import numpy as np
import os
import kagglehub

# Sample corpus (fit TF-IDF)
sample_corpus = [
    "I love machine learning.",
    "Deep learning models are powerful.",
    "Natural language processing is fascinating."
]

def load_models():
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(sample_corpus)

    # Download Word2Vec-SLIM model
    path = kagglehub.dataset_download("stoicstatic/word2vecslim300k")
    model_path = os.path.join(path, "GoogleNews-vectors-negative300-SLIM.bin")
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    return tfidf_vectorizer, w2v_model

def get_tfidf_embedding(text, tfidf_vectorizer):
    return tfidf_vectorizer.transform([text]).toarray()[0]

def get_w2v_embedding(text, w2v_model):
    words = text.lower().split()
    word_vecs = [w2v_model[word] for word in words if word in w2v_model]
    if not word_vecs:
        return np.zeros(w2v_model.vector_size)
    return np.mean(word_vecs, axis=0)