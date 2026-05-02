# 🗣️ Project 10 — Talk to Database (NL2SQL)

> **Training Project** | Python · LangChain · SQLite · Streamlit  
> Built during AI & Data Science training to build a natural language database interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![LLM](https://img.shields.io/badge/LLM-Text%20to%20SQL-blueviolet)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Domain](https://img.shields.io/badge/Domain-LLMs%20%2F%20Database-lightgrey)

---

## 📌 About This Project

This is the **final project** of my AI training — and it combines everything I learned about LLMs with real-world database interaction.

The idea is simple but powerful: **What if anyone could query a database just by asking a question in plain English?** No SQL knowledge needed. You type *"Show me all products with stock below 10"* and the system automatically generates and runs the correct SQL query, returning the results in a clean table.

This project uses LangChain and an LLM to translate natural language into SQL, then executes the query against a live SQLite database.

---

## ✨ What It Does

- 💬 **Ask questions in plain English** — no SQL required
- 🤖 **LLM generates SQL** — schema-aware query generation from natural language
- ⚡ **Executes queries** on a live SQLite / MySQL database
- 📊 **Displays results** in a clean, readable table format
- 🔒 **Schema-aware** — the model knows your table structure to avoid invalid queries
- 🌐 **Streamlit interface** — clean UI for asking questions and viewing results

---

## 🗂️ Project Structure

```
10.TALK_TO_DATABASE/
├── app.py            # Streamlit application
├── utils.py          # DB connection, query execution, LLM setup
├── requirements.txt  # Dependencies
└── README.md
```

### What each file does

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — text input for questions, table display for results |
| `utils.py` | DB connection, schema extraction, LangChain NL2SQL chain setup |
| `requirements.txt` | LangChain, SQLAlchemy, Streamlit and other dependencies |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd AI_training_with_10_projects/10.TALK_TO_DATABASE
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=sqlite:///your_database.db
```

### 4. Run the app
```bash
streamlit run app.py
```

Type a question like *"How many products are in stock?"* and see the magic! ✨

---

## 🧠 How It Works

```
User Question (natural language)
        ↓
LLM + DB Schema Context
        ↓
Generated SQL Query
        ↓
Database Execution (SQLite / MySQL)
        ↓
Results displayed in the UI
```

The key is providing the LLM with the **database schema** as context — it knows what tables and columns exist, so it generates valid, accurate SQL.

---

## 💬 Example Queries

| Question | Generated SQL |
|----------|--------------|
| Show me all products | `SELECT * FROM products` |
| Products with stock below 10 | `SELECT * FROM products WHERE stock < 10` |
| Total number of orders | `SELECT COUNT(*) FROM orders` |
| Top 5 most expensive items | `SELECT * FROM products ORDER BY price DESC LIMIT 5` |

---

## 🧠 What I Learned

- How LangChain's `SQLDatabaseChain` works for NL2SQL
- Why providing schema context to the LLM is critical for accuracy
- SQLAlchemy for database connections in Python
- Handling edge cases — invalid queries, empty results, error messages
- Building a polished end-to-end AI application with a real use case

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core logic |
| LangChain | NL2SQL chain orchestration |
| OpenAI / LLM | Natural language to SQL generation |
| SQLite | Local database |
| SQLAlchemy | Database ORM and connection |
| Streamlit | Web interface |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

> Part of [AI Training — 10 Projects](../README.md)

---

## 📄 License

[MIT License](../LICENSE)
