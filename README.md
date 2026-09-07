# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

A data science project developed for Parcl Co. Limited (via Unified Mentor) to identify hidden buyer segments in real estate client data using unsupervised machine learning, and to translate those segments into actionable investment profiles.

## Project Overview

Real estate companies interact with highly diverse buyers — individual home buyers, institutional investors, international buyers, high-net-worth investors, and first-time buyers. Without segmentation, companies treat all buyers the same, leading to inefficient marketing, poor targeting, and missed investment opportunities.

This project uses AI-based clustering to uncover natural buyer segments from client and property transaction data, then presents the findings through an interactive dashboard.

## Data

- **clients.csv** — 2,000 client records (demographics, acquisition purpose, loan status, referral channel, satisfaction score)
- **properties.csv** — 10,000 property transaction records, linked to clients via `client_ref`

Client-level investment features (total properties owned, total spend, average price per property) were engineered by aggregating property transactions per client and merging them into the client dataset.

## Methodology

1. **Data Cleaning** — handled mixed date formats, verified no missing values, removed currency formatting from price fields
2. **Feature Engineering** — aggregated property purchase history per client (total properties, total spend, average price)
3. **Feature Encoding** — one-hot encoding for low-cardinality categorical fields, label encoding for region/country
4. **Feature Scaling** — StandardScaler applied to all numeric features
5. **Clustering** — K-Means and Hierarchical (Agglomerative) clustering applied and cross-validated against each other
6. **Optimal Cluster Selection** — Elbow Method and Silhouette Score used to justify k=5
7. **Cluster Interpretation** — each segment profiled by spend, loan usage, geography, satisfaction, and demographics

## Buyer Segments Identified

| Segment | Size | Key Characteristics |
|---|---|---|
| Standard Buyers | 680 | Largest segment; lowest satisfaction and spend |
| High-Value Buyers | 578 | Highest average price per property |
| Loan-Dependent Buyers | 444 | Highest loan usage rate |
| International Buyers | 248 | Youngest average age; majority non-USA (Canada, France, Belgium) |
| Luxury Investors | 50 | Smallest segment; highest spend, most properties owned, highest satisfaction |

## Dashboard

An interactive Streamlit dashboard is included (`app.py`) with four modules:
1. Buyer Segmentation Overview — cluster distribution
2. Investor Behavior Dashboard — spend and loan patterns by segment
3. Geographic Buyer Analysis — segment composition by country
4. Segment Insights Panel — descriptive statistics per segment

Users can filter by country, region, acquisition purpose, and client type.

## Tech Stack

- Python (pandas, scikit-learn, matplotlib, scipy)
- Google Colab (data processing and clustering)
- Streamlit (dashboard)

## Files

- `app.py` — Streamlit dashboard application
- `clients_final.csv` — processed client dataset with cluster labels
- `requirements.txt` — Python package dependencies

## Author

Simran
