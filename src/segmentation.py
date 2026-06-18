import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


SEGMENT_LABELS = {
    0: "Champions",        # High R, F, M
    1: "Loyal Customers",  # High F, M; medium R
    2: "At-Risk",          # Used to buy well, but recency is high
    3: "Lost Cheap",       # Low across the board
    4: "Promising",        # Recent, low F
}

SEGMENT_COLORS = {
    "Champions": "#2ecc71",
    "Loyal Customers": "#3498db",
    "At-Risk": "#e67e22",
    "Lost Cheap": "#e74c3c",
    "Promising": "#9b59b6",
}


def run_kmeans_segmentation(rfm_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm_df[features].fillna(0)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Elbow + Silhouette to validate k=5
    inertias, sil_scores = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sc)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_sc, labels))

    print("Silhouette scores by k:", {k+2: round(s, 3) for k, s in enumerate(sil_scores)})

    # Final model
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df = rfm_df.copy()
    rfm_df["Cluster"] = km_final.fit_predict(X_sc)

    # Label clusters by RFM centroid ranking
    centers = pd.DataFrame(
        scaler.inverse_transform(km_final.cluster_centers_),
        columns=features
    )
    centers["cluster_id"] = range(n_clusters)
    # Rank: Champions = high F, M; low R
    centers["score"] = (
        -centers["Recency"] +
        centers["Frequency"] * 2 +
        centers["Monetary"] / centers["Monetary"].max() * 3
    )
    centers_ranked = centers.sort_values("score", ascending=False).reset_index(drop=True)
    label_map = {
        int(centers_ranked.loc[i, "cluster_id"]): label
        for i, label in enumerate(SEGMENT_LABELS.values())
    }
    rfm_df["Segment"] = rfm_df["Cluster"].map(label_map)

    joblib.dump(km_final, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/segmentation_scaler.pkl")

    # Print segment summary
    seg_summary = rfm_df.groupby("Segment")[features].median().round(1)
    seg_summary["Count"] = rfm_df["Segment"].value_counts()
    print("\n── Segment Summary (Median) ──")
    print(seg_summary.sort_values("Count", ascending=False))

    return rfm_df


if __name__ == "__main__":
    from src.feature_engineering import add_rfm_scores

    rfm = pd.read_csv("data/rfm_with_churn.csv")
    rfm = run_kmeans_segmentation(rfm)
    rfm.to_csv("data/rfm_segmented.csv", index=False)
    print("Saved data/rfm_segmented.csv")