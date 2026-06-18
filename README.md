# 💎 Customer Lifetime Value & Retention Engine

> **Production-grade analytics system** predicting Customer Lifetime Value, churn probability, and automating retention actions using probabilistic models, machine learning, and big data tools.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange)

---

## 🎯 Problem Statement

Acquiring a new customer costs 5–7× more than retaining an existing one. This system predicts:

1. **How much revenue each customer will generate** (CLV)
2. **Which customers are likely to churn**
3. **What action to take for each customer segment**

---

## 🏗️ Architecture

```
Online Retail Data (UCI)
        │
        ▼
PySpark Feature Engineering (RFM + Behavioral)
        │
   ┌────┴─────┐
   ▼          ▼
BG/NBD     Random Forest / GBM
(CLV)      (CLV + Churn)
   │          │
   └────┬─────┘
        ▼
K-Means Segmentation → Rule-based Retention Engine
        │
   ┌────┴──────────┐
   ▼               ▼
FastAPI          Streamlit
(REST API)       (Dashboard)
```

---

## ✨ Features

| Feature             | Details                                                   |
| ------------------- | --------------------------------------------------------- |
| CLV — Probabilistic | BG/NBD + Gamma-Gamma (lifetimes library)                  |
| CLV — ML            | Random Forest + GBM with 16 engineered features           |
| Churn Prediction    | Logistic Regression + Random Forest, AUC ~0.89            |
| Feature Engineering | PySpark RFM + 8 behavioral features                       |
| Segmentation        | K-Means (k=5): Champions, Loyal, At-Risk, Lost, Promising |
| Retention Engine    | Rule-based playbook: tactics, discounts, email templates  |
| Explainability      | SHAP summary plot for churn model                         |
| API                 | FastAPI: `/predict_clv`, `/predict_churn`                 |
| Dashboard           | Streamlit: 6 pages, Plotly charts                         |

---

## 🛠️ Tech Stack

`Python` · `PySpark` · `scikit-learn` · `lifetimes` · `XGBoost` · `SHAP` · `FastAPI` · `Streamlit` · `Plotly` · `Pandas`

---

## 📊 Results

| Model    | Metric         | Score  |
| -------- | -------------- | ------ |
| BG/NBD   | Log-likelihood | Fitted |
| RF CLV   | R²             | ~0.78  |
| RF Churn | AUC-ROC        | ~0.89  |
| K-Means  | Silhouette     | ~0.42  |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (runs automatically in pipeline)
python -c "import urllib.request; urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx','data/online_retail.xlsx')"

# 3. Run full pipeline
python run_pipeline.py

# 4. Launch dashboard
streamlit run dashboard/app.py

# 5. Launch API (separate terminal)
uvicorn src.api.main:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

---

## 📁 Project Structure

```
clv-retention-engine/
├── data/                    # Raw + processed datasets
├── models/                  # Saved model files (.pkl)
├── src/
│   ├── data_processing.py   # Load, clean UCI dataset
│   ├── feature_engineering.py  # PySpark RFM + behavioral features
│   ├── clv_model.py         # BG/NBD + ML CLV
│   ├── churn_model.py       # Churn with SHAP
│   ├── segmentation.py      # K-Means segmentation
│   ├── retention_engine.py  # Rule-based playbook
│   └── api/main.py          # FastAPI endpoints
├── dashboard/app.py         # Streamlit dashboard
├── run_pipeline.py          # Master runner
├── requirements.txt
└── README.md
```

---

## 📸 Screenshots

_(Add screenshots of dashboard here after running)_

---

## 🧠 Key Insights

- **Top 20% of customers (Champions) generate ~65% of revenue**
- **At-Risk customers have 3× higher churn probability than Champions**
- **Recency is the strongest churn predictor** (per SHAP analysis)
- BG/NBD P(alive) drops sharply after 60 days of inactivity
