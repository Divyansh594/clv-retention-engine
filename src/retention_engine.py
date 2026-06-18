import pandas as pd

RETENTION_PLAYBOOK = {
    "Champions": {
        "risk_level": "Low",
        "action": "Reward & Upsell",
        "tactics": [
            "Offer early access to new products",
            "Launch a VIP loyalty program with exclusive perks",
            "Ask for product reviews and referrals",
            "Upsell premium tiers or bundles",
        ],
        "email_subject": "🎉 You're one of our best customers — here's a special reward",
        "discount_pct": 5,
    },
    "Loyal Customers": {
        "risk_level": "Low-Medium",
        "action": "Strengthen Loyalty",
        "tactics": [
            "Send personalized thank-you offers",
            "Introduce subscription or membership benefits",
            "Cross-sell complementary product categories",
            "Feature them in customer spotlight campaigns",
        ],
        "email_subject": "💙 Thank you for sticking with us — a gift inside",
        "discount_pct": 10,
    },
    "At-Risk": {
        "risk_level": "High",
        "action": "Win-Back Campaign",
        "tactics": [
            "Send win-back email with time-limited offer",
            "Highlight new arrivals since last purchase",
            "Survey: why did you stop buying?",
            "Offer free shipping on next order",
        ],
        "email_subject": "⚠️ We miss you — here's 15% off to come back",
        "discount_pct": 15,
    },
    "Lost Cheap": {
        "risk_level": "Very High",
        "action": "Minimal Investment / Reactivate",
        "tactics": [
            "Low-cost bulk reactivation email",
            "Push only highest-margin products",
            "Suppress from expensive campaigns",
            "Final attempt: 20% off + free gift",
        ],
        "email_subject": "Last chance — we'd love to have you back 💌",
        "discount_pct": 20,
    },
    "Promising": {
        "risk_level": "Medium",
        "action": "Nurture & Convert",
        "tactics": [
            "Welcome series: educate on product range",
            "Bundle deals to increase order value",
            "Trigger behavioural emails after browsing",
            "Invite to loyalty program early",
        ],
        "email_subject": "🌟 You're just getting started — here's what you might love",
        "discount_pct": 12,
    },
}


def get_retention_recommendation(segment: str) -> dict:
    return RETENTION_PLAYBOOK.get(segment, {
        "risk_level": "Unknown",
        "action": "Manual Review",
        "tactics": ["Investigate customer history"],
        "email_subject": "We value your business",
        "discount_pct": 10,
    })


def generate_retention_report(rfm_segmented: pd.DataFrame) -> pd.DataFrame:
    df = rfm_segmented.copy()

    df["RetentionAction"] = df["Segment"].apply(
        lambda s: get_retention_recommendation(s)["action"]
    )
    df["RiskLevel"] = df["Segment"].apply(
        lambda s: get_retention_recommendation(s)["risk_level"]
    )
    df["RecommendedDiscount"] = df["Segment"].apply(
        lambda s: get_retention_recommendation(s)["discount_pct"]
    )
    df["EmailSubject"] = df["Segment"].apply(
        lambda s: get_retention_recommendation(s)["email_subject"]
    )

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/rfm_segmented.csv")
    df_out = generate_retention_report(df)
    df_out.to_csv("data/final_customer_data.csv", index=False)
    print(df_out[["CustomerID", "Segment", "RetentionAction", "RiskLevel", "RecommendedDiscount"]].head(10))
    print("Saved data/final_customer_data.csv")