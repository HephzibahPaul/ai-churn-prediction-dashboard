import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt  # For enterprise-grade charts

# Load full dataset for insights tab
DATA_PATH = Path("../data/raw_customers.csv")
df_full = None
if DATA_PATH.exists():
    df_full = pd.read_csv(DATA_PATH)


def load_model_and_features():
    model_path = Path("../models/churn_model.pkl")
    feature_cols_path = Path("../models/feature_cols.pkl")
    feature_importances_path = Path("../models/feature_importances.pkl")

    if not model_path.exists() or not feature_cols_path.exists():
        st.error("Model files not found. Please run `python -m src.train_model` from project root first.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(feature_cols_path, "rb") as f:
        feature_cols = pickle.load(f)

    # Load feature importances if present
    feature_importances = None
    if feature_importances_path.exists():
        with open(feature_importances_path, "rb") as f:
            feature_importances = pickle.load(f)

    return model, feature_cols, feature_importances


def preprocess_single_input(user_input: dict, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([user_input])
    # One-hot encode same as training
    df = pd.get_dummies(df, columns=["Gender", "Geography"], drop_first=True)
    # Align with training columns
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="centered"
)

st.markdown("#### 🏢 Customer Intelligence | Banking & Financial Services")
st.title("📊 Customer Churn Prediction & Insights Dashboard")
st.write(
    "Use this dashboard to estimate customer churn risk and support **data-driven retention decisions** "
    "through portfolio-level insights, segments, and key drivers."
)
st.markdown("---")

# ----------- Sidebar (Professional BA UI) -----------
st.sidebar.markdown("## 📘 Project Overview")
st.sidebar.markdown(
    """
    **Project:** AI-Powered Churn  
    **Role:** Business Analyst  
    **Model:** Random Forest  
    **Tech:** Python • Streamlit • sklearn • Altair  
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Navigation")

mode = st.sidebar.radio(
    "",
    ["🔮 Predict Single Customer", "📊 Churn Insights"]
)

model, feature_cols, feature_importances = load_model_and_features()

# -------------------- MODE 1: SINGLE CUSTOMER PREDICTION --------------------
if mode == "🔮 Predict Single Customer":
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        geography = st.selectbox("Geography", ["India", "Germany", "France", "Spain"])
        age = st.slider("Age", 18, 80, 35)
        tenure = st.slider("Tenure with Company (years)", 0, 10, 3)
        num_products = st.slider("Number of Products", 1, 4, 2)

    with col2:
        balance = st.number_input("Account Balance", min_value=0.0, value=60000.0, step=1000.0)
        has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", min_value=10000.0, value=80000.0, step=5000.0)

    user_input = {
        "Gender": gender,
        "Geography": geography,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumProducts": num_products,
        "HasCreditCard": 1 if has_credit_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0,
        "EstimatedSalary": estimated_salary
    }

    if st.button("Predict Churn Risk"):
        X_single = preprocess_single_input(user_input, feature_cols)
        churn_proba = model.predict_proba(X_single)[0, 1]
        churn_percent = churn_proba * 100

        st.markdown("---")
        st.subheader("Prediction Result")
        st.write(f"**Estimated Churn Probability:** `{churn_percent:.2f}%`")

        # Risk band
        if churn_percent >= 70:
            risk_label = "High"
            st.error("⚠ High churn risk. Immediate retention action recommended.")
        elif churn_percent >= 40:
            risk_label = "Medium"
            st.warning("⚠ Medium churn risk. Monitor and engage proactively.")
        else:
            risk_label = "Low"
            st.success("✅ Low churn risk. Customer is likely to stay.")

        # Suggested actions
        st.markdown("### Suggested Business Actions")
        if risk_label == "High":
            st.markdown(
                """
                - Personal call from relationship manager  
                - Strong retention offers (15–20% discount)  
                - Analyze complaints / service history  
                """
            )
        elif risk_label == "Medium":
            st.markdown(
                """
                - Engagement email campaigns  
                - Loyalty discount  
                - Feedback survey  
                """
            )
        else:
            st.markdown(
                """
                - Cross-sell opportunities  
                - Invite to referral program  
                - Continue engagement email flows  
                """
            )

        # --- LTV + retention budget estimation ---
        st.markdown("### 💰 Financial Impact (Approximate)")
        ltv = estimated_salary * 0.15 * (tenure + 1) / 12
        retention_budget = ltv * 0.2

        st.write(f"**Estimated LTV:** ~₹{ltv:,.0f}")
        st.write(f"**Suggested Max Retention Spend:** ~₹{retention_budget:,.0f}")
        st.caption("Business Analysts often evaluate customer LTV before recommending retention actions.")

# -------------------- MODE 2: CHURN INSIGHTS --------------------
else:
    st.subheader("📊 Portfolio-Level Churn Insights")

    if df_full is None:
        st.warning("Full dataset not found.")
    else:
        # Overall churn rate
        churn_rate = df_full["Churn"].mean() * 100
        st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

        # ----------- Churn by Geography (Sapphire Blue) -----------
        st.markdown("### Churn by Geography")
        geo_churn = df_full.groupby("Geography")["Churn"].mean().sort_values() * 100
        geo_df = geo_churn.reset_index()
        geo_df.columns = ["Geography", "ChurnRate"]

        geo_chart = (
            alt.Chart(geo_df)
            .mark_bar(color="#2980B9")  # Sapphire Blue
            .encode(
                x=alt.X("Geography:N", sort="-y"),
                y=alt.Y("ChurnRate:Q"),
            )
        )
        st.altair_chart(geo_chart, use_container_width=True)

        # ----------- Churn by Age Group (Emerald Green) -----------
        st.markdown("### Churn by Age Group")
        bins = [18, 25, 35, 45, 55, 65, 80]
        labels = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
        df_full["AgeGroup"] = pd.cut(df_full["Age"], bins=bins, labels=labels, right=False)
        age_churn = df_full.groupby("AgeGroup")["Churn"].mean().sort_values() * 100
        age_df = age_churn.reset_index()
        age_df.columns = ["AgeGroup", "ChurnRate"]

        age_chart = (
            alt.Chart(age_df)
            .mark_bar(color="#2ECC71")  # Emerald Green
            .encode(
                x=alt.X("AgeGroup:N", sort=labels),
                y=alt.Y("ChurnRate:Q"),
            )
        )
        st.altair_chart(age_chart, use_container_width=True)

        # ----------- Customer Segments (Coral Red) -----------
        st.markdown("### Customer Segments (Rule-Based)")
        df_seg = df_full.copy()
        df_seg["Segment"] = "New / Neutral"

        df_seg.loc[(df_seg["Churn"] == 0) & (df_seg["Tenure"] >= 5), "Segment"] = "Loyal"
        df_seg.loc[df_seg["Churn"] == 1, "Segment"] = "At-Risk"

        high_balance_threshold = df_seg["Balance"].median()
        df_seg.loc[
            (df_seg["Churn"] == 1) & (df_seg["Balance"] > high_balance_threshold),
            "Segment"
        ] = "High-Value At-Risk"

        seg_counts = df_seg["Segment"].value_counts()
        seg_df = seg_counts.reset_index()
        seg_df.columns = ["Segment", "Count"]

        seg_chart = (
            alt.Chart(seg_df)
            .mark_bar(color="#E74C3C")  # Coral Red
            .encode(
                x=alt.X("Segment:N", sort="-y"),
                y=alt.Y("Count:Q"),
            )
        )
        st.altair_chart(seg_chart, use_container_width=True)

        st.markdown("### Sample of High-Value At-Risk Customers")
        hv = df_seg[df_seg["Segment"] == "High-Value At-Risk"].head(10)

        # Keep only required columns & reset index
        hv_reset = hv[["CustomerID", "Geography", "Age", "Tenure", "Balance", "NumProducts"]].reset_index(drop=True)

        # Add clean serial numbers
        hv_reset.index = hv_reset.index + 1

        st.dataframe(hv_reset)


        # ----------- Drivers of Churn (Royal Gold) -----------
        if feature_importances is not None:
            st.markdown("### 🔍 Key Drivers of Churn (Feature Importance)")
            fi_df = pd.DataFrame(
                {"feature": list(feature_importances.keys()),
                 "importance": list(feature_importances.values())}
            ).sort_values("importance", ascending=False)

            top_n = st.slider("Show top N drivers", 3, min(10, len(fi_df)), 5)
            fi_top = fi_df.head(top_n)

            drivers_chart = (
                alt.Chart(fi_top)
                .mark_bar(color="#F1C40F")  # Royal Gold
                .encode(
                    x=alt.X("importance:Q", title="Importance"),
                    y=alt.Y("feature:N", sort="-x"),
                )
            )
            st.altair_chart(drivers_chart, use_container_width=True)

            st.caption("Feature importance helps identify what drives churn in the model.")
