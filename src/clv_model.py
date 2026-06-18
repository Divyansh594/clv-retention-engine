import pandas as pd
import numpy as np
import joblib
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# PROBABILISTIC CLV: BG/NBD + Gamma-Gamma
# ─────────────────────────────────────────────

def prepare_bgf_summary(df_clean: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Lifetimes requires: frequency, recency, T (age of customer)."""
    summary = summary_data_from_transaction_data(
        df_clean,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=snapshot_date,
        freq="D"
    )
    # lifetimes returns recency/T in chosen freq (days)
    return summary


def fit_bgnbd(summary: pd.DataFrame) -> BetaGeoFitter:
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(
        summary["frequency"],
        summary["recency"],
        summary["T"]
    )
    print(f"BG/NBD fitted. Params: {bgf.params_}")
    return bgf


def fit_gamma_gamma(summary: pd.DataFrame) -> GammaGammaFitter:
    # Only customers with repeat purchases
    returning = summary[summary["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(returning["frequency"], returning["monetary_value"])
    print(f"Gamma-Gamma fitted. Params: {ggf.params_}")
    return ggf


def predict_probabilistic_clv(
    summary: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    time_horizon: int = 180,  # days
    discount_rate: float = 0.01
) -> pd.DataFrame:
    summary = summary.copy()

    # Expected number of purchases in next `time_horizon` days
    summary["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        time_horizon,
        summary["frequency"],
        summary["recency"],
        summary["T"]
    )

    # P(alive)
    summary["prob_alive"] = bgf.conditional_probability_alive(
        summary["frequency"],
        summary["recency"],
        summary["T"]
    )

    # Expected CLV (only for returning customers)
    returning_mask = summary["frequency"] > 0
    summary.loc[returning_mask, "CLV_Probabilistic"] = ggf.customer_lifetime_value(
        bgf,
        summary.loc[returning_mask, "frequency"],
        summary.loc[returning_mask, "recency"],
        summary.loc[returning_mask, "T"],
        summary.loc[returning_mask, "monetary_value"],
        time=time_horizon / 30,  # in months
        discount_rate=discount_rate / 30
    )

    # For one-time buyers: estimate as avg monetary × predicted purchases
    avg_monetary = summary.loc[returning_mask, "monetary_value"].mean()
    summary.loc[~returning_mask, "CLV_Probabilistic"] = (
        summary.loc[~returning_mask, "predicted_purchases"] * avg_monetary
    )

    summary["CLV_Probabilistic"] = summary["CLV_Probabilistic"].clip(lower=0)
    return summary


# ─────────────────────────────────────────────
# ML CLV: Random Forest + GBM
# ─────────────────────────────────────────────

def create_clv_target(rfm_df: pd.DataFrame, df_clean: pd.DataFrame,
                      snapshot_date: pd.Timestamp, future_days: int = 180) -> pd.DataFrame:
    """
    For ML CLV: use first 12 months as features, next 6 months revenue as target.
    Simplified: use total Monetary as proxy target (for single-period datasets).
    """
    # Split: first 75% of timeline = features, last 25% = target
    split_date = df_clean["InvoiceDate"].min() + \
        (df_clean["InvoiceDate"].max() - df_clean["InvoiceDate"].min()) * 0.75

    future_revenue = df_clean[df_clean["InvoiceDate"] > split_date].groupby("CustomerID")["Revenue"].sum().reset_index()
    future_revenue.columns = ["CustomerID", "CLV_Target"]

    merged = rfm_df.merge(future_revenue, on="CustomerID", how="left")
    merged["CLV_Target"] = merged["CLV_Target"].fillna(0)
    return merged


def train_ml_clv(df_with_target: pd.DataFrame):
    features = [
        "Recency", "Frequency", "Monetary", "AvgOrderValue", "StdOrderValue",
        "TotalTransactions", "UniqueProducts", "TotalQuantity",
        "ActiveMonths", "LifespanDays", "FreqPerMonth",
        "AvgInterPurchaseDays", "R_Score", "F_Score", "M_Score", "RFM_Score"
    ]

    X = df_with_target[features].fillna(0)
    y = df_with_target["CLV_Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)

    gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gbm.fit(X_train_sc, y_train)

    for name, model in [("RandomForest", rf), ("GBM", gbm)]:
        preds = model.predict(X_test_sc)
        print(f"{name} — MAE: {mean_absolute_error(y_test, preds):.2f} | R²: {r2_score(y_test, preds):.3f}")

    # Save best model (RF)
    joblib.dump(rf, "models/clv_rf_model.pkl")
    joblib.dump(scaler, "models/clv_scaler.pkl")
    joblib.dump(features, "models/clv_features.pkl")

    print("CLV ML models saved.")
    return rf, gbm, scaler, features, X_test_sc, y_test


if __name__ == "__main__":
    from src.data_processing import load_data, clean_data, get_snapshot_date
    from src.feature_engineering import compute_rfm_spark, add_rfm_scores
    import os
    os.makedirs("models", exist_ok=True)

    df = clean_data(load_data())
    snapshot = get_snapshot_date(df)

    # Probabilistic
    summary = prepare_bgf_summary(df, snapshot)
    bgf = fit_bgnbd(summary)
    ggf = fit_gamma_gamma(summary)
    clv_prob = predict_probabilistic_clv(summary, bgf, ggf)
    clv_prob.to_csv("data/clv_probabilistic.csv")

    joblib.dump(bgf, "models/bgf.pkl")
    joblib.dump(ggf, "models/ggf.pkl")

    # ML
    rfm = add_rfm_scores(pd.read_csv("data/rfm_features.csv"))
    df_target = create_clv_target(rfm, df, snapshot)
    train_ml_clv(df_target)

    print("CLV pipeline complete.")