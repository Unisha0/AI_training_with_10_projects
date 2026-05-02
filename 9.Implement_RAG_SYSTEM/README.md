# 🔍 RAG System — Retrieval-Augmented Generation Chatbot

> A Retrieval-Augmented Generation (RAG) system that lets you chat with your own PDF documents using LLMs — powered by LangChain, vector stores, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![LangChain](https://img.shields.io/badge/LangChain-RAG-green) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project implements a full **Retrieval-Augmented Generation (RAG)** pipeline that allows users to upload PDF documents and ask natural language questions about them. The system retrieves relevant chunks from the document and passes them to an LLM to generate accurate, context-grounded answers.

---

## ✨ Features

- 📄 Upload and parse PDF documents
- 🧩 Intelligent document chunking and embedding
- 🔎 Vector-based semantic search (FAISS / Chroma)
- 🤖 LLM-powered answer generation with context
- 💬 Streamlit chat interface
- 📓 Jupyter Notebook walkthrough included

---

## 🗂️ Project Structure

```
9.Implement_RAG_SYSTEM/
├── app.py                      # Streamlit chatbot app
├── utils.py                    # RAG pipeline utilities
├── Ch13-Chatbot-RAG_pdf.ipynb  # Notebook walkthrough
├── requirements.txt            # Dependencies
├── assets/                     # UI assets
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd AI_training_with_10_projects/9.Implement_RAG_SYSTEM

pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

---

## 🧠 RAG Pipeline

```
PDF Upload
    ↓
Document Chunking (RecursiveCharacterTextSplitter)
    ↓
Embedding (OpenAI / HuggingFace)
    ↓
Vector Store (FAISS / Chroma)
    ↓
User Query → Semantic Search → Top-K Chunks
    ↓
LLM (GPT / Llama) + Context → Answer
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core logic |
| LangChain | RAG orchestration |
| FAISS / Chroma | Vector database |
| OpenAI / HuggingFace | LLM & Embeddings |
| Streamlit | Chat UI |
| PyPDF2 | PDF parsing |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

---

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
