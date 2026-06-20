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

<img width="1843" height="1071" alt="image" src="https://github.com/user-attachments/assets/e74f3a22-dade-4177-98c5-3a6373508275" />
<img width="1428" height="769" alt="image" src="https://github.com/user-attachments/assets/36039c9d-cc5b-41a9-a8b5-d374e8cfe4f6" />
<img width="1431" height="791" alt="image" src="https://github.com/user-attachments/assets/87262fae-f160-449a-a5ef-5a028fc21290" />
<img width="1409" height="626" alt="image" src="https://github.com/user-attachments/assets/4523fdc0-46b8-4483-b488-0ce13f8446fe" />
<img width="1425" height="919" alt="image" src="https://github.com/user-attachments/assets/7eedd9e3-b330-4296-b31d-ca21edc9ffb6" />
<img width="1434" height="876" alt="image" src="https://github.com/user-attachments/assets/8f18e5c5-f3d8-4419-80d8-d37b710428f4" />
<img width="1403" height="617" alt="image" src="https://github.com/user-attachments/assets/6d9d01cf-4352-49d9-9e80-fd516a444ace" />
<img width="1425" height="683" alt="image" src="https://github.com/user-attachments/assets/e91e1936-2b64-4590-ba84-58e531af3d99" />
<img width="1426" height="683" alt="image" src="https://github.com/user-attachments/assets/bd6a1a3a-5fdb-44fc-ad4e-c03684a3c1b8" />
<img width="1429" height="947" alt="image" src="https://github.com/user-attachments/assets/a73dd978-04ae-46c6-8da7-a6e1c976bf3f" />
<img width="1424" height="961" alt="image" src="https://github.com/user-attachments/assets/83548722-f0c1-4e6b-b9cf-4d5fa2e045e0" />



---

## 🧠 Key Insights

- **Top 20% of customers (Champions) generate ~65% of revenue**
- **At-Risk customers have 3× higher churn probability than Champions**
- **Recency is the strongest churn predictor** (per SHAP analysis)
- BG/NBD P(alive) drops sharply after 60 days of inactivity
