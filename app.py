import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Page setup ----------
st.set_page_config(page_title="Parcl Buyer Segmentation", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("clients_final.csv")
    cluster_names = {
        0: "Standard Buyers",
        1: "International Buyers",
        2: "High-Value Buyers",
        3: "Loan-Dependent Buyers",
        4: "Luxury Investors"
    }
    df["segment"] = df["cluster"].map(cluster_names)
    return df

df = load_data()

# Consistent color palette across all charts (same as research paper)
SEGMENT_COLORS = {
    "Standard Buyers": "#4C6EF5",
    "High-Value Buyers": "#63B3ED",
    "Loan-Dependent Buyers": "#38A169",
    "International Buyers": "#DD6B20",
    "Luxury Investors": "#B83280"
}

# ---------- Title ----------
st.title("Parcl Buyer Segmentation & Investment Profiling")
st.caption("Machine learning based buyer segmentation for real estate market intelligence")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

country_options = ["All"] + sorted(df["country"].unique().tolist())
selected_country = st.sidebar.selectbox("Country", country_options)

region_options = ["All"] + sorted(df["region"].unique().tolist())
selected_region = st.sidebar.selectbox("Region", region_options)

purpose_options = ["All", "Home", "Investment"]
selected_purpose = st.sidebar.selectbox(
    "Acquisition Purpose",
    purpose_options
)

client_type_options = ["All", "Individual", "Company"]
selected_client_type = st.sidebar.selectbox("Client Type", client_type_options)

# ---------- Apply filters ----------
filtered = df.copy()

if selected_country != "All":
    filtered = filtered[filtered["country"] == selected_country]

if selected_region != "All":
    filtered = filtered[filtered["region"] == selected_region]

if selected_purpose != "All":
    col = f"acquisition_purpose_{selected_purpose}"
    if col in filtered.columns:
        filtered = filtered[filtered[col] == True]

if selected_client_type != "All":
    if selected_client_type == "Individual":
        filtered = filtered[filtered["client_type_Individual"] == True]
    else:
        filtered = filtered[filtered["client_type_Individual"] == False]

st.sidebar.markdown(f"**{len(filtered)}** clients match current filters")

# ---------- Module 1: Buyer Segmentation Overview ----------
st.header("1. Buyer Segmentation Overview")

col1, col2 = st.columns([2, 1])

with col1:
    segment_counts = filtered["segment"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [SEGMENT_COLORS[s] for s in segment_counts.index]
    ax.bar(segment_counts.index, segment_counts.values, color=colors)
    ax.set_ylabel("Number of Clients")
    ax.set_title("Cluster Distribution")
    plt.xticks(rotation=15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.metric("Total Clients (filtered)", len(filtered))
    st.metric("Segments Represented", filtered["segment"].nunique())
    if len(filtered) > 0:
        top_segment = filtered["segment"].value_counts().idxmax()
        st.metric("Largest Segment", top_segment)

# ---------- Module 1B: Financial Impact Overview ----------
st.header("1B. Financial Impact Overview")

total_portfolio_value = filtered["total_spend"].sum()
revenue_by_segment = filtered.groupby("segment")["total_spend"].sum().sort_values(ascending=False)
revenue_pct_by_segment = (revenue_by_segment / total_portfolio_value * 100).round(1)

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Total Portfolio Value (filtered)", f"${total_portfolio_value:,.0f}")
with kpi2:
    top_revenue_segment = revenue_by_segment.idxmax()
    st.metric("Top Revenue-Generating Segment", top_revenue_segment)
with kpi3:
    top_revenue_share = revenue_pct_by_segment.max()
    st.metric(f"{top_revenue_segment} Share of Total Revenue", f"{top_revenue_share}%")

col_rev1, col_rev2 = st.columns(2)

with col_rev1:
    fig_rev, ax_rev = plt.subplots(figsize=(6, 4.5))
    colors_rev = [SEGMENT_COLORS[s] for s in revenue_by_segment.index]
    ax_rev.barh(revenue_by_segment.index, revenue_by_segment.values, color=colors_rev)
    ax_rev.set_xlabel("Total Revenue Contribution ($)")
    ax_rev.set_title("Revenue Contribution by Segment")
    ax_rev.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_rev)

with col_rev2:
    fig_pct, ax_pct = plt.subplots(figsize=(6, 4.5))
    colors_pct = [SEGMENT_COLORS[s] for s in revenue_pct_by_segment.sort_values(ascending=False).index]
    ax_pct.pie(revenue_pct_by_segment.sort_values(ascending=False).values,
               labels=revenue_pct_by_segment.sort_values(ascending=False).index,
               autopct='%1.1f%%', colors=colors_pct, startangle=90,
               textprops={'fontsize': 8})
    ax_pct.set_title("Share of Total Revenue by Segment")
    plt.tight_layout()
    st.pyplot(fig_pct)

st.caption(
    "This module highlights the financial concentration of buyer segments: a small, high-value segment can "
    "represent a disproportionate share of total revenue relative to its size, which is critical for prioritizing "
    "retention and relationship management resources."
)

# ---------- Module 2: Investor Behavior Dashboard ----------
st.header("2. Investor Behavior Dashboard")

col3, col4 = st.columns(2)

with col3:
    avg_spend = filtered.groupby("segment")["total_spend"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    colors2 = [SEGMENT_COLORS[s] for s in avg_spend.index]
    ax2.barh(avg_spend.index, avg_spend.values, color=colors2)
    ax2.set_xlabel("Average Total Spend ($)")
    ax2.set_title("Average Spend by Segment")
    ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

with col4:
    loan_rate = filtered.groupby("segment")["loan_applied_Yes"].mean().sort_values(ascending=False) * 100
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    colors3 = [SEGMENT_COLORS[s] for s in loan_rate.index]
    ax3.barh(loan_rate.index, loan_rate.values, color=colors3)
    ax3.set_xlabel("% Applied for Loan")
    ax3.set_title("Loan Usage by Segment")
    ax3.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)

# ---------- Module 3: Geographic Buyer Analysis ----------
st.header("3. Geographic Buyer Analysis")

top_countries = filtered["country"].value_counts().head(10)
fig4, ax4 = plt.subplots(figsize=(10, 4.5))
ax4.bar(top_countries.index, top_countries.values, color="#4C6EF5")
ax4.set_ylabel("Number of Clients")
ax4.set_title("Top 10 Countries by Client Count")
plt.xticks(rotation=30, ha="right")
ax4.grid(axis="y", alpha=0.3)
plt.tight_layout()
st.pyplot(fig4)

st.subheader("Segment Composition by Country (Top 5 Countries)")
top5 = filtered["country"].value_counts().head(5).index
geo_table = pd.crosstab(filtered[filtered["country"].isin(top5)]["country"],
                         filtered[filtered["country"].isin(top5)]["segment"])
st.dataframe(geo_table, use_container_width=True)

# ---------- Module 4: Segment Insights Panel ----------
st.header("4. Segment Insights Panel")

insights = filtered.groupby("segment").agg(
    avg_age=("age", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
    avg_properties=("total_properties", "mean"),
    avg_spend=("total_spend", "mean"),
    avg_price=("avg_price", "mean"),
    pct_loan=("loan_applied_Yes", "mean"),
    client_count=("client_id", "count")
).round(2)

insights["pct_loan"] = (insights["pct_loan"] * 100).round(1)
insights.columns = ["Avg Age", "Avg Satisfaction", "Avg Properties Owned",
                     "Avg Total Spend ($)", "Avg Price per Property ($)",
                     "% Loan Applied", "Number of Clients"]

st.dataframe(insights, use_container_width=True)

st.caption("Data source: Parcl Co. Limited client and property transaction records, processed via K-Means clustering (k=5).")
