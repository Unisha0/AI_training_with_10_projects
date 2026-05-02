# 🤖 AI Training — 10 Hands-On Projects

> Built by **Unisha Chaulagain** during an intensive AI & Data Science training program.  
> A complete learning journey from classical ML to Generative AI, LLMs, and RAG systems.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Projects-10-brightgreen" />
  <img src="https://img.shields.io/badge/Topics-ML%20%7C%20DL%20%7C%20NLP%20%7C%20GenAI%20%7C%20LLMs-orange" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
  <img src="https://img.shields.io/badge/Author-Unisha%20Chaulagain-blueviolet" />
</p>

---

## 👩‍💻 About This Repository

This repo documents my hands-on learning journey through **10 real-world AI and Data Science projects**. Each project was built from scratch during training sessions, covering a wide range of topics — from building a simple inventory dashboard, training neural networks, working with Generative AI models, all the way to building a full RAG-based PDF chatbot and a natural language database query system.

Every project is self-contained with its own:
- 📁 Source code
- 📦 `requirements.txt`
- 📖 Detailed `README.md`

---

## 📂 Project Index

| # | Project | Domain | Key Tech |
|---|---------|--------|----------|
| 1 | 🗂️ [Inventory Management System](./1.INVENTORY_SYSTEM/) | Data App | Python, Streamlit, Pandas |
| 2 | 💬 [Sentiment Analysis](./2.SENTIMENT_ANALYSIS/) | NLP / ML | NLTK, scikit-learn, Jupyter |
| 3 | 🖼️ [CNN Image Classification — MNIST](./3.CNN_IMAGE/) | Deep Learning | PyTorch, CNN, Streamlit |
| 4 | 🎨 [GAN & DCGAN — Mini Image Generator](./4.TRAIN_GAN_AND_DCGAN(MINI%20_IMAGE_AI)/) | Generative AI | PyTorch, GAN, DCGAN |
| 5 | 🐾 [CGAN — Pet Image Detector](./5.TRAIN_CGAN(Pet_image_detector)/) | Generative AI | PyTorch, Conditional GAN |
| 6 | 📐 [Text Embedding Techniques](./6.Explore_varoious_text_embedding%20techniques/) | NLP | Word2Vec, GloVe, BERT, HuggingFace |
| 7 | 🔧 Fine-Tune GPT-2 | LLMs | GPT-2, HuggingFace Transformers |
| 8 | 🦙 [Local LLMs with Ollama](./8.SIZE%28%3C%3D5GB%29MODELS_WITH_OLLAMA_PROJECTS/) | LLMs | Ollama, Mistral, Llama3, Jupyter |
| 9 | 🔍 [RAG System — PDF Chatbot](./9.Implement_RAG_SYSTEM/) | LLMs / RAG | LangChain, FAISS, Streamlit |
| 10 | 🗣️ [Talk to Database — NL2SQL](./10.TALK_TO_DATABASE/) | LLMs / DB | LangChain, SQLite, Streamlit |

---

## 🗂️ Project 1 — Inventory Management System

A fully functional inventory dashboard built with **Python and Streamlit**. Supports adding, updating, and deleting products, with real-time stock visualization and low-stock alerts.

- 📊 Interactive charts and KPI metrics
- 📁 CSV-based data persistence
- 🔔 Low stock threshold alerts

📖 [View README](./1.INVENTORY_SYSTEM/README.md)

---

## 💬 Project 2 — Sentiment Analysis

An NLP pipeline that processes raw text data and classifies sentiment as **positive, negative, or neutral** using classical machine learning models.

- 🧹 Full text preprocessing (tokenization, stopwords, lemmatization)
- 📈 Model training with accuracy, F1-score evaluation
- 📓 Jupyter Notebook for exploration

📖 [View README](./2.SENTIMENT_ANALYSIS/README.md)

---

## 🖼️ Project 3 — CNN Image Classification (MNIST)

A **Convolutional Neural Network** trained from scratch on the MNIST handwritten digits dataset, achieving ~99% test accuracy. Served through a Streamlit web app for live predictions.

- 🧠 Custom CNN architecture (conv layers, pooling, dropout, FC)
- 🌐 Streamlit app for drawing/uploading digits
- 🗂️ Modular code — model, training, utils separated

📖 [View README](./3.CNN_IMAGE/README.md)

---

## 🎨 Project 4 — GAN & DCGAN (Mini Image AI)

Implementation of both a **Vanilla GAN** and a **Deep Convolutional GAN (DCGAN)** for generating synthetic images. Demonstrates the full adversarial training loop between Generator and Discriminator networks.

- ⚔️ Adversarial training (Generator vs Discriminator)
- 📉 Loss tracking and convergence visualization
- 🖼️ Image generation from random latent vectors

📖 [View README](./4.TRAIN_GAN_AND_DCGAN(MINI%20_IMAGE_AI)/README.md)

---

## 🐾 Project 5 — CGAN Pet Image Detector

A **Conditional GAN (CGAN)** that conditions both the Generator and Discriminator on class labels, enabling controlled generation of specific pet categories (cats, dogs). Includes a full inference pipeline.

- 🏷️ Label-conditioned image generation
- 💾 Model checkpointing and reloading
- 🔮 Inference script for generating new images

📖 [View README](./5.TRAIN_CGAN(Pet_image_detector)/README.md)

---

## 📐 Project 6 — Text Embedding Techniques

A comprehensive comparison of **6 text embedding methods** — from classical TF-IDF to transformer-based BERT embeddings. Visualizes how different representations capture semantic meaning.

| Method | Type |
|--------|------|
| TF-IDF | Statistical |
| Word2Vec | Shallow Neural |
| GloVe | Pre-trained Static |
| FastText | Subword-based |
| BERT | Transformer |
| Sentence Transformers | Semantic Similarity |

📖 [View README](./6.Explore_varoious_text_embedding%20techniques/README.md)

---

## 🔧 Project 7 — Fine-Tune GPT-2

Fine-tuning **GPT-2** on small custom datasets for domain-specific text generation. Explores how a pre-trained language model can be adapted to generate personalized content with minimal data.

> 📌 *Code coming soon*

---

## 🦙 Project 8 — Local LLMs with Ollama (≤5GB Models)

Running and experimenting with **open-source LLMs entirely offline** using Ollama. Tests models like Mistral, Llama3, Gemma, and Phi-3 — all under 5GB — with zero API cost and full privacy.

- 🔌 Zero internet dependency after model download
- 📓 Jupyter Notebook with prompt experiments
- ⚡ Fast local inference on consumer hardware

📖 [View README](./8.SIZE%28%3C%3D5GB%29MODELS_WITH_OLLAMA_PROJECTS/README.md)

---

## 🔍 Project 9 — RAG System (PDF Chatbot)

A complete **Retrieval-Augmented Generation (RAG)** pipeline that lets users upload PDF files and ask natural language questions. Combines semantic search over document chunks with LLM-generated answers.

```
PDF → Chunking → Embeddings → Vector Store (FAISS)
                                      ↓
              User Query → Semantic Search → Top Chunks → LLM → Answer
```

📖 [View README](./9.Implement_RAG_SYSTEM/README.md)

---

## 🗣️ Project 10 — Talk to Database (NL2SQL)

An AI-powered interface that translates **plain English questions into SQL queries** and executes them against a live database — no SQL knowledge needed.

- 💬 *"Show me all products with stock below 10"* → generates & runs SQL instantly
- 🗄️ SQLite database integration
- 🔒 Schema-aware query generation

📖 [View README](./10.TALK_TO_DATABASE/README.md)

---

## 🛠️ Full Tech Stack

```
Languages      →  Python 3.8+
Deep Learning  →  PyTorch, TensorFlow
NLP            →  NLTK, spaCy, HuggingFace Transformers, Gensim
LLMs           →  LangChain, Ollama, OpenAI API, GPT-2
Generative AI  →  GAN, DCGAN, CGAN
Vector Search  →  FAISS, ChromaDB
Databases      →  SQLite, SQLAlchemy
UI / Apps      →  Streamlit
Notebooks      →  Jupyter
```

---

## ⚙️ How to Run Any Project

```bash
# 1. Clone the repo
git clone https://github.com/Unisha0/AI_training_with_10_projects.git

# 2. Go into any project folder
cd AI_training_with_10_projects/9.Implement_RAG_SYSTEM

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

> 💡 Each project folder has its own `requirements.txt` — install only what you need.

---

## 📈 Learning Path

```
Week 1  →  Inventory System (Python basics, Streamlit)
Week 2  →  Sentiment Analysis (NLP, ML)
Week 3  →  CNN (Deep Learning, image classification)
Week 4  →  GAN / DCGAN / CGAN (Generative AI)
Week 5  →  Text Embeddings (NLP representations)
Week 6  →  GPT-2 Fine-tuning (LLMs)
Week 7  →  Ollama Local LLMs (offline AI)
Week 8  →  RAG System (LLM + retrieval)
Week 9  →  Talk to Database (NL2SQL)
```

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

---

## 📄 License

This repository is licensed under the [MIT License](./LICENSE).

---

> ⭐ If you found this helpful or inspiring, feel free to star the repo!
