# 🗣️ Talk to Database — Natural Language to SQL

> An AI-powered application that translates natural language questions into SQL queries, enabling anyone to query a database without knowing SQL.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![LLM](https://img.shields.io/badge/LLM-Text%20to%20SQL-blueviolet) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project builds a **Natural Language to SQL (NL2SQL)** interface using a large language model (LLM). Users can type plain English questions (e.g., *"Show me all orders from last month"*) and the system automatically generates and executes the corresponding SQL query, returning results in a clean interface.

---

## ✨ Features

- 💬 Ask questions in plain English
- 🤖 LLM-powered SQL query generation
- 🗄️ Execute queries on a live SQLite/MySQL database
- 📊 Display query results in a readable table format
- 🔒 Schema-aware query generation to prevent errors

---

## 🗂️ Project Structure

```
10.TALK_TO_DATABASE/
├── app.py           # Streamlit application
├── utils.py         # DB connection & query execution
├── requirements.txt # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd AI_training_with_10_projects/10.TALK_TO_DATABASE

pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=sqlite:///your_database.db
```

### Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

```
User Question (natural language)
        ↓
LLM + DB Schema Context
        ↓
Generated SQL Query
        ↓
Database Execution
        ↓
Results displayed in the UI
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core logic |
| LangChain / OpenAI | NL2SQL generation |
| SQLite / SQLAlchemy | Database layer |
| Streamlit | Web interface |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

---

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
