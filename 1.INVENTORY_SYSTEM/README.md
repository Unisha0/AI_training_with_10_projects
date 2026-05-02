# 🗂️ Project 1 — Inventory Management System

> **Training Project** | Python · Streamlit · Pandas  
> Built during AI & Data Science training to learn Python app development and data visualization.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Data%20App-lightgrey)

---

## 📌 About This Project

This was my **first hands-on project** during training — focused on building a real-world application using Python. The goal was to create a fully working inventory management tool with a visual dashboard that any business could actually use.

I built a system that tracks products, manages stock levels, raises low-stock alerts, and presents everything in a clean Streamlit dashboard with charts and KPIs.

---

## ✨ What It Does

- 📦 **Add / Update / Delete** products from the inventory
- 📊 **Dashboard view** with charts showing stock levels and KPIs
- 🔍 **Search and filter** products by name or category
- 🔔 **Low stock alerts** when quantity drops below a threshold
- 💾 **CSV-based storage** so data persists between sessions
- 📈 **Real-time metrics** — total products, total stock value, low stock count

---

## 🗂️ Project Structure

```
1.INVENTORY_SYSTEM/
├── app.py           # Main Streamlit app — handles product CRUD
├── dashboard.py     # Dashboard UI — charts, KPIs, summaries
└── README.md
```

### What each file does

| File | Purpose |
|------|---------|
| `app.py` | Core app logic — add/edit/delete products, form handling |
| `dashboard.py` | Visual dashboard — bar charts, pie charts, stock metrics |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd AI_training_with_10_projects/1.INVENTORY_SYSTEM
```

### 2. Install dependencies
```bash
pip install streamlit pandas
```

### 3. Launch the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser — the dashboard loads automatically!

---

## 🧠 What I Learned

- Building multi-page Streamlit apps
- Working with Pandas DataFrames for CRUD operations
- Creating data visualizations with Streamlit charts
- Structuring a Python project cleanly with separate files

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Core application logic |
| Streamlit | Latest | Web dashboard and UI |
| Pandas | Latest | Data storage and manipulation |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

> Part of [AI Training — 10 Projects](../README.md)

---

## 📄 License

[MIT License](../LICENSE)
