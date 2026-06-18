import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="CLV & Retention Engine",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Data ──
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "final_customer_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_clv_prob():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "clv_probabilistic.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

df = load_data()
clv_prob = load_clv_prob()

SEGMENT_COLORS = {
    "Champions": "#2ecc71",
    "Loyal Customers": "#3498db",
    "At-Risk": "#e67e22",
    "Lost Cheap": "#e74c3c",
    "Promising": "#9b59b6",
}

# ── Sidebar ──
st.sidebar.image("https://img.icons8.com/fluency/96/customer-insight.png", width=60)
st.sidebar.title("CLV & Retention Engine")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "💰 CLV Analysis", "⚠️ Churn Analysis",
     "🎯 Segmentation", "🔧 Retention Engine", "🔍 Customer Lookup"]
)

if df is None:
    st.error("⚠️ No data found. Run the full pipeline first (see README).")
    st.stop()

# ── Page: Overview ──
if page == "📊 Overview":
    st.title("📊 Executive Dashboard")
    st.markdown("*Customer Lifetime Value & Retention Analytics Platform*")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Avg CLV (ML)", f"£{df['CLV_Target'].mean():.0f}" if "CLV_Target" in df.columns else "N/A")
    col3.metric("Avg Churn Risk", f"{df['churn_prob'].mean():.1%}" if "churn_prob" in df.columns else "N/A")
    col4.metric("Champions", f"{(df['Segment'] == 'Champions').sum():,}")
    col5.metric("At-Risk", f"{(df['Segment'] == 'At-Risk').sum():,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        seg_counts = df["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(
            seg_counts, values="Count", names="Segment",
            title="Customer Segment Distribution",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "churn_prob" in df.columns:
            fig2 = px.histogram(
                df, x="churn_prob", nbins=30,
                title="Churn Probability Distribution",
                color_discrete_sequence=["#e74c3c"]
            )
            fig2.update_layout(xaxis_title="Churn Probability", yaxis_title="# Customers")
            st.plotly_chart(fig2, use_container_width=True)

    # Revenue by segment
    st.markdown("### 💰 Revenue by Segment")
    seg_rev = df.groupby("Segment")["Monetary"].sum().reset_index().sort_values("Monetary", ascending=False)
    fig3 = px.bar(
        seg_rev, x="Segment", y="Monetary",
        color="Segment", color_discrete_map=SEGMENT_COLORS,
        title="Total Revenue by Segment"
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Page: CLV Analysis ──
elif page == "💰 CLV Analysis":
    st.title("💰 Customer Lifetime Value Analysis")

    if clv_prob is not None:
        col1, col2 = st.columns(2)
        with col1:
            clv_prob_clean = clv_prob[clv_prob["CLV_Probabilistic"] < clv_prob["CLV_Probabilistic"].quantile(0.99)]
            fig = px.histogram(
                clv_prob_clean, x="CLV_Probabilistic", nbins=40,
                title="BG/NBD + Gamma-Gamma CLV Distribution",
                color_discrete_sequence=["#3498db"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                clv_prob_clean.head(500),
                x="frequency", y="CLV_Probabilistic",
                color="prob_alive", size="monetary_value",
                title="CLV vs Frequency (coloured by P(Alive))",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🏆 Top 20 Highest CLV Customers")
        top20 = clv_prob.nlargest(20, "CLV_Probabilistic")[
            ["frequency", "recency", "monetary_value", "prob_alive",
             "predicted_purchases", "CLV_Probabilistic"]
        ].round(2)
        st.dataframe(top20, use_container_width=True)

    # ML vs Probabilistic comparison
    if "CLV_Target" in df.columns:
        st.markdown("### 🔬 ML CLV Distribution")
        fig_ml = px.histogram(
            df[df["CLV_Target"] < df["CLV_Target"].quantile(0.99)],
            x="CLV_Target", nbins=40,
            title="ML Model CLV Target Distribution",
            color_discrete_sequence=["#2ecc71"]
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        st.info("""
        **BG/NBD vs ML Comparison:**
        - **BG/NBD + Gamma-Gamma**: Probabilistic, interpretable, works with sparse data. 
          Outputs P(alive) and expected transactions. Best for marketing teams.
        - **ML (Random Forest)**: Captures non-linear patterns, feature interactions. 
          Better predictive accuracy with rich feature sets. Best for product/engineering teams.
        """)


# ── Page: Churn Analysis ──
elif page == "⚠️ Churn Analysis":
    st.title("⚠️ Churn Risk Analysis")

    if "churn_prob" in df.columns:
        col1, col2, col3 = st.columns(3)
        col1.metric("High Risk (>70%)", f"{(df['churn_prob'] > 0.7).sum():,}",
                    delta=f"{(df['churn_prob'] > 0.7).mean():.1%} of base")
        col2.metric("Medium Risk (40-70%)", f"{((df['churn_prob'] > 0.4) & (df['churn_prob'] <= 0.7)).sum():,}")
        col3.metric("Low Risk (<40%)", f"{(df['churn_prob'] <= 0.4).sum():,}")

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.box(
                df, x="Segment", y="churn_prob",
                color="Segment", color_discrete_map=SEGMENT_COLORS,
                title="Churn Probability by Segment"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.scatter(
                df.sample(min(1000, len(df))),
                x="Recency", y="churn_prob",
                color="Segment", color_discrete_map=SEGMENT_COLORS,
                title="Churn Probability vs Recency"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🚨 High-Risk Customers (Top 20 by Churn Probability)")
        high_risk = df.nlargest(20, "churn_prob")[
            ["CustomerID", "Segment", "Recency", "Frequency", "Monetary",
             "churn_prob", "RetentionAction"]
        ].round(3)
        st.dataframe(high_risk, use_container_width=True)

        # SHAP image
        shap_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "churn_shap_summary.png"
        )
        if os.path.exists(shap_path):
            st.markdown("### 🔍 SHAP Feature Importance")
            st.image(shap_path, caption="SHAP Values — Top features driving churn prediction")


# ── Page: Segmentation ──
elif page == "🎯 Segmentation":
    st.title("🎯 Customer Segmentation")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter_3d(
            df.sample(min(2000, len(df))),
            x="Recency", y="Frequency", z="Monetary",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            title="3D RFM Segmentation",
            opacity=0.7,
            size_max=5
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_stats = df.groupby("Segment").agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Avg_Churn_Prob=("churn_prob", "mean")
        ).round(1).reset_index()
        st.dataframe(seg_stats, use_container_width=True)

    # Radar chart per segment
    st.markdown("### 📡 Segment Profiles (Radar Chart)")
    metrics = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts"]
    seg_radar = df.groupby("Segment")[metrics].mean()
    seg_radar_norm = (seg_radar - seg_radar.min()) / (seg_radar.max() - seg_radar.min())

    fig_radar = go.Figure()
    for seg in seg_radar_norm.index:
        values = seg_radar_norm.loc[seg].tolist()
        values += [values[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=seg,
            line_color=SEGMENT_COLORS.get(seg, "#888")
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig_radar, use_container_width=True)


# ── Page: Retention Engine ──
elif page == "🔧 Retention Engine":
    st.title("🔧 Retention Engine & Playbook")

    for seg, playbook in {
        "Champions": SEGMENT_COLORS,
        "Loyal Customers": SEGMENT_COLORS,
        "At-Risk": SEGMENT_COLORS,
        "Lost Cheap": SEGMENT_COLORS,
        "Promising": SEGMENT_COLORS,
    }.items():
        from src.retention_engine import get_retention_recommendation
        rec = get_retention_recommendation(seg)
        count = (df["Segment"] == seg).sum()
        with st.expander(f"**{seg}** — {count} customers | Risk: {rec['risk_level']} | Discount: {rec['discount_pct']}%"):
            st.markdown(f"**📧 Email Subject:** `{rec['email_subject']}`")
            st.markdown(f"**🎯 Action:** {rec['action']}")
            st.markdown("**📋 Tactics:**")
            for t in rec["tactics"]:
                st.markdown(f"- {t}")


# ── Page: Customer Lookup ──
elif page == "🔍 Customer Lookup":
    st.title("🔍 Individual Customer Insights")

    customer_ids = df["CustomerID"].unique().tolist()
    selected_id = st.selectbox("Select Customer ID", sorted(customer_ids))

    if selected_id:
        customer = df[df["CustomerID"] == selected_id].iloc[0]
        from src.retention_engine import get_retention_recommendation
        rec = get_retention_recommendation(customer.get("Segment", "Unknown"))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Segment", customer.get("Segment", "N/A"))
        col2.metric("Churn Probability", f"{customer.get('churn_prob', 0):.1%}")
        col3.metric("Recency (days)", f"{customer.get('Recency', 0):.0f}")
        col4.metric("Total Revenue", f"£{customer.get('Monetary', 0):.0f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 📊 Customer Profile")
            profile_data = {
                "Frequency": customer.get("Frequency", 0),
                "Avg Order Value": f"£{customer.get('AvgOrderValue', 0):.2f}",
                "Unique Products": customer.get("UniqueProducts", 0),
                "Active Months": customer.get("ActiveMonths", 0),
                "RFM Score": customer.get("RFM_Score", 0),
                "Risk Level": rec["risk_level"],
            }
            st.table(pd.Series(profile_data).rename("Value"))

        with col_b:
            st.markdown("### 🎯 Retention Recommendation")
            st.success(f"**Action:** {rec['action']}")
            st.info(f"**Email:** {rec['email_subject']}")
            st.warning(f"**Discount:** {rec['discount_pct']}% recommended")
            st.markdown("**Tactics:**")
            for t in rec["tactics"]:
                st.markdown(f"✅ {t}")