# 🦙 Local LLM Projects with Ollama (≤5GB Models)

> Running and experimenting with local large language models (LLMs) under 5GB using Ollama — no internet or API keys required.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project demonstrates how to run **open-source LLMs locally** using [Ollama](https://ollama.com). By focusing on models ≤5GB in size (e.g., Llama 3, Mistral, Gemma, Phi-3), this project makes AI accessible on standard consumer hardware — no GPU cloud, no API key needed.

---

## ✨ Features

- 🦙 Run models like `llama3`, `mistral`, `gemma`, `phi3` locally
- 📓 Jupyter Notebook for interactive LLM experimentation
- 💬 Prompt engineering and response comparison
- 🔌 Ollama Python API integration
- 🏃 Zero cloud dependency — fully offline AI

---

## 🗂️ Project Structure

```
8.SIZE(<=5GB)MODELS_WITH_OLLAMA_PROJECTS/
├── test_ollama.ipynb   # Jupyter Notebook — LLM experiments
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

1. **Install Ollama**: Download from [ollama.com](https://ollama.com/download)
2. **Pull a model** (e.g., Mistral ~4GB):

```bash
ollama pull mistral
```

### Installation

```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd "AI_training_with_10_projects/8.SIZE(<=5GB)MODELS_WITH_OLLAMA_PROJECTS"

pip install ollama jupyter
```

### Run the Notebook

```bash
jupyter notebook test_ollama.ipynb
```

---

## 🧠 Supported Models (≤5GB)

| Model | Size | Use Case |
|-------|------|----------|
| Mistral 7B (Q4) | ~4.1GB | General purpose |
| Llama 3 8B (Q4) | ~4.7GB | Reasoning |
| Gemma 2B | ~1.7GB | Lightweight |
| Phi-3 Mini | ~2.3GB | Code & reasoning |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Ollama | Local LLM runtime |
| Python | Scripting & API calls |
| Jupyter Notebook | Interactive experiments |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

---

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
