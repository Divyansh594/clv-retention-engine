import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def define_churn_label(rfm_df: pd.DataFrame, churn_threshold_days: int = 90) -> pd.DataFrame:
    """
    Business definition: Customer is churned if Recency > 90 days.
    Adjust threshold based on your domain.
    """
    df = rfm_df.copy()
    df["Churned"] = (df["Recency"] > churn_threshold_days).astype(int)
    print(f"Churn rate: {df['Churned'].mean():.1%}")
    return df


def train_churn_models(rfm_with_churn: pd.DataFrame):
    features = [
        "Frequency", "Monetary", "AvgOrderValue", "StdOrderValue",
        "TotalTransactions", "UniqueProducts", "TotalQuantity",
        "ActiveMonths", "LifespanDays", "FreqPerMonth",
        "AvgInterPurchaseDays", "R_Score", "F_Score", "M_Score", "RFM_Score"
    ]

    X = rfm_with_churn[features].fillna(0)
    y = rfm_with_churn["Churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Logistic Regression ──
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_preds = lr.predict_proba(X_test_sc)[:, 1]
    print(f"Logistic Regression — AUC: {roc_auc_score(y_test, lr_preds):.3f}")
    print(classification_report(y_test, (lr_preds > 0.5).astype(int)))

    # ── Random Forest ──
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)
    rf.fit(X_train_sc, y_train)
    rf_preds = rf.predict_proba(X_test_sc)[:, 1]
    print(f"Random Forest — AUC: {roc_auc_score(y_test, rf_preds):.3f}")
    print(classification_report(y_test, (rf_preds > 0.5).astype(int)))

    # ── SHAP Explainability ──
    print("Computing SHAP values (may take ~1 min)...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_sc[:200])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values[1], X_test_sc[:200],
        feature_names=features, show=False
    )
    plt.tight_layout()
    plt.savefig("models/churn_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP plot saved to models/churn_shap_summary.png")

    # Save
    joblib.dump(rf, "models/churn_rf_model.pkl")
    joblib.dump(lr, "models/churn_lr_model.pkl")
    joblib.dump(scaler, "models/churn_scaler.pkl")
    joblib.dump(features, "models/churn_features.pkl")

    # Attach predictions to df
    rfm_with_churn = rfm_with_churn.copy()
    rfm_with_churn["churn_prob"] = np.nan
    # Predict on all data
    X_all_sc = scaler.transform(X.fillna(0))
    rfm_with_churn["churn_prob"] = rf.predict_proba(X_all_sc)[:, 1]

    return rf, lr, scaler, features, rfm_with_churn


if __name__ == "__main__":
    rfm = pd.read_csv("data/rfm_features.csv")
    from src.feature_engineering import add_rfm_scores
    rfm = add_rfm_scores(rfm)
    rfm_churn = define_churn_label(rfm)
    rf, lr, scaler, features, rfm_out = train_churn_models(rfm_churn)
    rfm_out.to_csv("data/rfm_with_churn.csv", index=False)
    print("Saved data/rfm_with_churn.csv")