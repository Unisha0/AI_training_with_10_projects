# 💬 Project 2 — Sentiment Analysis

> **Training Project** | Python · NLP · scikit-learn · Jupyter  
> Built during AI & Data Science training to learn Natural Language Processing and text classification.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Domain](https://img.shields.io/badge/Domain-NLP%20%2F%20ML-lightgrey)

---

## 📌 About This Project

This was my **second training project** — my introduction to Natural Language Processing. The goal was to understand how machines can read and understand human text, specifically to detect whether a piece of text is **positive, negative, or neutral**.

I built a full ML pipeline from raw text all the way to predictions, and also explored the process interactively through a Jupyter Notebook.

---

## ✨ What It Does

- 🧹 **Text preprocessing pipeline** — tokenization, stopword removal, lemmatization
- 📊 **Exploratory Data Analysis** — word frequency, class distribution, visualizations
- 🤖 **Trains a classification model** — using Naive Bayes / Logistic Regression
- 📈 **Evaluates performance** — accuracy, precision, recall, F1-score
- 📓 **Jupyter Notebook** for step-by-step exploration and experimentation

---

## 🗂️ Project Structure

```
2.SENTIMENT_ANALYSIS/
├── sentiment_analysis.py   # Full ML pipeline — preprocessing + training + evaluation
├── txt.ipynb               # Jupyter Notebook for interactive exploration
├── requirements.txt        # Dependencies
└── README.md
```

### What each file does

| File | Purpose |
|------|---------|
| `sentiment_analysis.py` | End-to-end script — load data, preprocess, train, evaluate |
| `txt.ipynb` | Notebook — explore data, visualize, experiment with models |
| `requirements.txt` | All required Python packages |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Unisha0/AI_training_with_10_projects.git
cd AI_training_with_10_projects/2.SENTIMENT_ANALYSIS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script or open the notebook
```bash
# Option A — run the pipeline
python sentiment_analysis.py

# Option B — explore interactively
jupyter notebook txt.ipynb
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~90%+ |
| F1-Score | ~0.89 |

---

## 🧠 What I Learned

- How NLP text preprocessing works (tokenization, stemming, lemmatization)
- Difference between Bag-of-Words and TF-IDF representations
- Training and evaluating text classifiers with scikit-learn
- How to read and interpret a confusion matrix and classification report
- Using Jupyter Notebooks for exploratory data analysis

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core logic |
| NLTK / spaCy | NLP preprocessing |
| scikit-learn | ML model training and evaluation |
| Pandas | Data manipulation |
| Jupyter Notebook | Interactive exploration |

---

## 👩‍💻 Author

**Unisha Chaulagain**  
[![GitHub](https://img.shields.io/badge/GitHub-Unisha0-black?logo=github)](https://github.com/Unisha0)

> Part of [AI Training — 10 Projects](../README.md)

---

## 📄 License

[MIT License](../LICENSE)
