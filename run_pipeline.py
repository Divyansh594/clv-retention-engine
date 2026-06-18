"""
Master pipeline runner. Run this to train all models end-to-end.
Usage: python run_pipeline.py
"""
import os
import pandas as pd

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("=" * 60)
print("STEP 1/6: Loading & Cleaning Data")
print("=" * 60)
from src.data_processing import load_data, clean_data, get_snapshot_date
df = clean_data(load_data())
snapshot = get_snapshot_date(df)
print(f"✅ Clean dataset: {df.shape[0]:,} rows | Snapshot: {snapshot.date()}")

print("\n" + "=" * 60)
print("STEP 2/6: Feature Engineering (PySpark RFM)")
print("=" * 60)
from src.feature_engineering import compute_rfm_spark, add_rfm_scores
rfm = compute_rfm_spark(df, snapshot)
rfm = add_rfm_scores(rfm)
rfm.to_csv("data/rfm_features.csv", index=False)
print(f"✅ RFM computed: {rfm.shape[0]:,} customers | {rfm.shape[1]} features")

print("\n" + "=" * 60)
print("STEP 3/6: CLV Models (BG/NBD + ML)")
print("=" * 60)
import joblib
from src.clv_model import (prepare_bgf_summary, fit_bgnbd, fit_gamma_gamma,
                            predict_probabilistic_clv, create_clv_target, train_ml_clv)

summary = prepare_bgf_summary(df, snapshot)
bgf = fit_bgnbd(summary)
ggf = fit_gamma_gamma(summary)
clv_prob = predict_probabilistic_clv(summary, bgf, ggf)
clv_prob.to_csv("data/clv_probabilistic.csv")
joblib.dump(bgf, "models/bgf.pkl")
joblib.dump(ggf, "models/ggf.pkl")

df_target = create_clv_target(rfm, df, snapshot)
train_ml_clv(df_target)
df_target.to_csv("data/rfm_with_clv.csv", index=False)
print("✅ CLV models trained and saved")

print("\n" + "=" * 60)
print("STEP 4/6: Churn Prediction")
print("=" * 60)
from src.churn_model import define_churn_label, train_churn_models
rfm_churn = define_churn_label(df_target)
_, _, _, _, rfm_churn_out = train_churn_models(rfm_churn)
rfm_churn_out.to_csv("data/rfm_with_churn.csv", index=False)
print("✅ Churn models trained and saved")

print("\n" + "=" * 60)
print("STEP 5/6: Customer Segmentation")
print("=" * 60)
from src.segmentation import run_kmeans_segmentation
rfm_seg = run_kmeans_segmentation(rfm_churn_out)
rfm_seg.to_csv("data/rfm_segmented.csv", index=False)
print("✅ Segmentation complete")

print("\n" + "=" * 60)
print("STEP 6/6: Retention Engine")
print("=" * 60)
from src.retention_engine import generate_retention_report
final_df = generate_retention_report(rfm_seg)
final_df.to_csv("data/final_customer_data.csv", index=False)
print(f"✅ Retention report generated for {len(final_df):,} customers")

print("\n" + "=" * 60)
print("🎉 PIPELINE COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("  📊 Dashboard:  streamlit run dashboard/app.py")
print("  🚀 API:        uvicorn src.api.main:app --reload --port 8000")
print("  📖 API docs:   http://localhost:8000/docs")