# 📐 Exploring Various Text Embedding Techniques

> A comprehensive exploration of modern and classical text embedding methods including TF-IDF, Word2Vec, GloVe, FastText, and transformer-based embeddings (BERT/SentenceTransformers).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![NLP](https://img.shields.io/badge/NLP-Text%20Embeddings-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project systematically explores and compares various **text embedding techniques** used in NLP. From classical bag-of-words approaches to state-of-the-art transformer embeddings, this project demonstrates how different representations capture semantic meaning and how they perform in downstream tasks.

---

## ✨ Techniques Covered

| Technique | Type |
|-----------|------|
| TF-IDF | Statistical |
| Word2Vec | Shallow Neural |
| GloVe | Pre-trained Static |
| FastText | Subword-based |
| BERT Embeddings | Transformer-based |
| Sentence Transformers | Semantic Similarity |

---

## 🗂️ Project Structure

```
6.Explore_varoious_text_embedding techniques/
├── app.py           # Streamlit comparison app
├── utils.py         # Embedding utility functions
├── requirements.txt # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd "AI_training_with_10_projects/6.Explore_varoious_text_embedding techniques"

pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

---

## 🧠 Concept

Text embeddings convert words/sentences into numerical vectors. This project compares:
- **Similarity scores** between different embedding methods
- **Visualization** of embedding spaces using PCA/t-SNE
- **Downstream task performance** comparison

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core logic |
| Gensim | Word2Vec / GloVe / FastText |
| HuggingFace Transformers | BERT embeddings |
| sentence-transformers | Semantic embeddings |
| Streamlit | Comparison UI |
| scikit-learn | TF-IDF |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

---

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
