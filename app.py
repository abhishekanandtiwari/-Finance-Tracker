"""
Personal Finance Tracker — Streamlit App
=========================================
Upload bank/credit-card transactions (CSV), get them auto-categorized using
a rule-based engine + an optional trainable ML classifier, explore spending
insights, and correct categories to make the ML model smarter over time.

Run with:  streamlit run app.py
"""

import io
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from categorizer import (
    categorize_dataframe,
    rule_based_categorize,
    MLCategorizer,
    CATEGORY_RULES,
    DEFAULT_CATEGORY,
)

st.set_page_config(page_title="Personal Finance Tracker", page_icon="💰", layout="wide")

ALL_CATEGORIES = sorted(list(CATEGORY_RULES.keys()) + [DEFAULT_CATEGORY])

# ----------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------
if "transactions" not in st.session_state:
    st.session_state.transactions = None
if "ml_model" not in st.session_state:
    st.session_state.ml_model = MLCategorizer()
    st.session_state.ml_model.load()  # load previously trained model if it exists


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    # Try to map common column name variants to a standard schema
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("date", "transaction date", "posted date"):
            rename_map[c] = "Date"
        elif cl in ("description", "memo", "narration", "details", "merchant"):
            rename_map[c] = "Description"
        elif cl in ("amount", "transaction amount", "value"):
            rename_map[c] = "Amount"
    df = df.rename(columns=rename_map)

    required = {"Date", "Description", "Amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected columns like Date, Description, Amount."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Date", "Amount"])
    return df


# ----------------------------------------------------------------------
# Sidebar — upload & controls
# ----------------------------------------------------------------------
st.sidebar.title("💰 Finance Tracker")
st.sidebar.markdown("Upload your transactions to get started.")

uploaded_file = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
use_sample = st.sidebar.button("Use sample data instead")

if uploaded_file is not None:
    try:
        raw_df = load_csv(uploaded_file)
        cat_df = categorize_dataframe(raw_df, ml_model=st.session_state.ml_model)
        st.session_state.transactions = cat_df
        st.sidebar.success(f"Loaded {len(cat_df)} transactions.")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

if use_sample:
    raw_df = load_csv("sample_transactions.csv")
    cat_df = categorize_dataframe(raw_df, ml_model=st.session_state.ml_model)
    st.session_state.transactions = cat_df
    st.sidebar.success(f"Loaded {len(cat_df)} sample transactions.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Expected CSV columns: **Date, Description, Amount** "
    "(negative amounts = expenses, positive = income)."
)

ml_status = "Trained ✅" if st.session_state.ml_model.is_trained() else "Not trained yet"
st.sidebar.markdown(f"**ML model status:** {ml_status}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("Personal Finance Tracker")

if st.session_state.transactions is None:
    st.info("👈 Upload a CSV of transactions, or click **Use sample data** in the sidebar to explore the dashboard.")
    st.markdown("""
    **How it works**
    1. Upload a CSV with `Date`, `Description`, `Amount` columns.
    2. Transactions are auto-categorized using keyword rules, refined by a
       machine-learning classifier (TF-IDF + Logistic Regression) once you've
       corrected a few categories.
    3. Review and fix any miscategorized transactions in the table below —
       each correction helps retrain the model.
    4. Explore spending insights and visual breakdowns.
    """)
    st.stop()

df = st.session_state.transactions.copy()

# ---------------- Filters ----------------
with st.expander("🔍 Filters", expanded=False):
    col1, col2, col3 = st.columns(3)
    date_min, date_max = df["Date"].min(), df["Date"].max()
    with col1:
        date_range = st.date_input("Date range", (date_min, date_max))
    with col2:
        selected_categories = st.multiselect(
            "Categories", sorted(df["Category"].unique()), default=sorted(df["Category"].unique())
        )
    with col3:
        txn_type = st.radio("Transaction type", ["All", "Expenses only", "Income only"], horizontal=True)

mask = (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
mask &= df["Category"].isin(selected_categories)
if txn_type == "Expenses only":
    mask &= df["Amount"] < 0
elif txn_type == "Income only":
    mask &= df["Amount"] > 0
fdf = df[mask].copy()

# ---------------- Key metrics ----------------
total_income = fdf.loc[fdf["Amount"] > 0, "Amount"].sum()
total_expenses = -fdf.loc[fdf["Amount"] < 0, "Amount"].sum()
net = total_income - total_expenses
avg_daily_spend = (
    total_expenses / max((fdf["Date"].max() - fdf["Date"].min()).days, 1)
    if len(fdf) > 0 else 0
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Income", f"₹{total_income:,.2f}")
m2.metric("Total Expenses", f"₹{total_expenses:,.2f}")
m3.metric("Net Savings", f"₹{net:,.2f}", delta=f"{(net/total_income*100 if total_income else 0):.1f}% of income")
m4.metric("Avg. Daily Spend", f"₹{avg_daily_spend:,.2f}")

st.markdown("---")

# ---------------- Visual dashboard ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Spending Breakdown", "📈 Trends Over Time", "🧠 Insights", "📝 Review & Correct"])

with tab1:
    expenses_df = fdf[fdf["Amount"] < 0].copy()
    expenses_df["Amount"] = expenses_df["Amount"].abs()
    cat_summary = expenses_df.groupby("Category")["Amount"].sum().sort_values(ascending=False).reset_index()

    c1, c2 = st.columns([1, 1])
    with c1:
        fig_pie = px.pie(cat_summary, names="Category", values="Amount",
                          title="Spending by Category", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_bar = px.bar(cat_summary, x="Amount", y="Category", orientation="h",
                          title="Spending by Category (Amount)", text_auto=".2s")
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(cat_summary.rename(columns={"Amount": "Total Spent"}), use_container_width=True)

with tab2:
    daily = fdf.copy()
    daily["Day"] = daily["Date"].dt.date
    daily_summary = daily.groupby("Day").apply(
        lambda g: pd.Series({
            "Income": g.loc[g["Amount"] > 0, "Amount"].sum(),
            "Expenses": -g.loc[g["Amount"] < 0, "Amount"].sum(),
        })
    ).reset_index()
    daily_summary["Net"] = daily_summary["Income"] - daily_summary["Expenses"]
    daily_summary["Cumulative Net"] = daily_summary["Net"].cumsum()

    fig_line = px.line(daily_summary, x="Day", y=["Income", "Expenses"],
                        title="Daily Income vs Expenses", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

    fig_cum = px.area(daily_summary, x="Day", y="Cumulative Net",
                       title="Cumulative Net Savings Over Time")
    st.plotly_chart(fig_cum, use_container_width=True)

    monthly = fdf.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
    monthly_cat = monthly[monthly["Amount"] < 0].groupby(["Month", "Category"])["Amount"].sum().abs().reset_index()
    fig_stack = px.bar(monthly_cat, x="Month", y="Amount", color="Category",
                        title="Monthly Spending by Category (Stacked)")
    st.plotly_chart(fig_stack, use_container_width=True)

with tab3:
    st.subheader("Automated Insights")
    insights = []

    if not cat_summary.empty:
        top_cat = cat_summary.iloc[0]
        insights.append(
            f"🏆 Your highest spending category is **{top_cat['Category']}** "
            f"at **₹{top_cat['Amount']:,.2f}**, "
            f"({top_cat['Amount']/total_expenses*100:.1f}% of total expenses)."
        )

    if len(daily_summary) >= 2:
        # compare first half vs second half of period for a simple trend signal
        half = len(daily_summary) // 2
        first_half_spend = daily_summary.iloc[:half]["Expenses"].sum()
        second_half_spend = daily_summary.iloc[half:]["Expenses"].sum()
        if first_half_spend > 0:
            change = (second_half_spend - first_half_spend) / first_half_spend * 100
            direction = "increased" if change > 0 else "decreased"
            insights.append(f"📉 Spending {direction} by **{abs(change):.1f}%** in the second half of the selected period compared to the first half.")

    if total_income > 0:
        savings_rate = net / total_income * 100
        if savings_rate < 0:
            insights.append("⚠️ You're spending more than you're earning in this period — consider reviewing discretionary categories.")
        elif savings_rate < 10:
            insights.append(f"💡 Your savings rate is **{savings_rate:.1f}%** — financial experts often recommend aiming for 20%+.")
        else:
            insights.append(f"✅ Nice! Your savings rate is **{savings_rate:.1f}%** for this period.")

    uncategorized_pct = (fdf["Category"] == DEFAULT_CATEGORY).mean() * 100
    if uncategorized_pct > 5:
        insights.append(
            f"🔖 **{uncategorized_pct:.1f}%** of transactions are uncategorized. "
            "Correct a few in the 'Review & Correct' tab to improve auto-categorization."
        )

    if not expenses_df.empty:
        biggest_txn = expenses_df.loc[expenses_df["Amount"].idxmax()]
        insights.append(
            f"💸 Largest single expense: **₹{biggest_txn['Amount']:,.2f}** "
            f"on *{biggest_txn['Description']}* ({biggest_txn['Date'].date()})."
        )

    for i in insights:
        st.markdown(f"- {i}")

    if not insights:
        st.write("Not enough data yet for insights — try uploading more transactions.")

with tab4:
    st.subheader("Review & correct categories")
    st.caption(
        "Fix any wrong categories below, then click **Retrain model** — "
        "your corrections teach the ML classifier to recognize similar transactions automatically next time."
    )

    edited_df = st.data_editor(
        fdf[["Date", "Description", "Amount", "Category"]].reset_index(drop=True),
        column_config={
            "Category": st.column_config.SelectboxColumn("Category", options=ALL_CATEGORIES, required=True),
        },
        use_container_width=True,
        num_rows="fixed",
        key="editor",
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("💾 Save corrections"):
            # write corrections back into the master transactions table by matching index
            full = st.session_state.transactions
            for i, row in edited_df.iterrows():
                match = (
                    (full["Date"] == row["Date"]) &
                    (full["Description"] == row["Description"]) &
                    (full["Amount"] == row["Amount"])
                )
                full.loc[match, "Category"] = row["Category"]
            st.session_state.transactions = full
            st.success("Corrections saved.")

    with colB:
        if st.button("🧠 Retrain ML model on corrected data"):
            full = st.session_state.transactions
            ok = st.session_state.ml_model.train(full["Description"].tolist(), full["Category"].tolist())
            if ok:
                st.session_state.ml_model.save()
                st.success("Model retrained and saved! It will now help categorize future uploads.")
            else:
                st.warning("Not enough labeled variety yet — correct a few more transactions across different categories, then try again.")

st.markdown("---")
st.download_button(
    "⬇️ Download categorized transactions as CSV",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="categorized_transactions.csv",
    mime="text/csv",
)
