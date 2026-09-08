# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

A financial analytics project developed for Parcl Co. Limited (via Unified Mentor) that applies unsupervised machine learning to segment real estate buyers and translate those segments into portfolio-level financial insight — not just descriptive clusters.

## Project Overview

Real estate companies interact with highly diverse buyers — individual home buyers, institutional investors, international buyers, high-net-worth investors, and first-time buyers. Without segmentation, companies treat all buyers the same, leading to inefficient marketing, poor targeting, and missed investment opportunities.

This project goes a step further than a purely technical clustering exercise: every segment identified is evaluated not only on its demographic composition, but on its **financial contribution to Parcl's overall client portfolio** — where revenue is concentrated, which relationships carry disproportionate value, and where financing behavior differs enough to warrant distinct commercial strategies.

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
8. **Financial Impact Analysis** — each segment's share of total portfolio revenue compared to its share of clients, to identify disproportionately valuable segments

## Buyer Segments Identified

| Segment | Size | % of Clients | % of Revenue | Revenue/Client Ratio | Key Characteristics |
|---|---|---|---|---|---|
| Standard Buyers | 680 | 34.0% | 28.7% | 0.84x | Lowest satisfaction and spend |
| High-Value Buyers | 578 | 28.9% | 34.8% | 1.20x | Highest average price per property |
| Loan-Dependent Buyers | 444 | 22.2% | 20.3% | 0.92x | Highest loan usage rate |
| International Buyers | 248 | 12.4% | 11.4% | 0.92x | Youngest; majority non-USA (Canada, France, Belgium) |
| Luxury Investors | 50 | 2.5% | 4.8% | **1.92x** | Highest spend, most properties owned, highest satisfaction |

**Key finding:** Luxury Investors represent only 2.5% of the client base but generate 4.8% of Parcl's total portfolio revenue (~$2.52B) — a 1.92x revenue-to-client ratio, the highest of any segment. This disproportionate financial concentration is the central business insight of the analysis and directly informs the retention priorities recommended in the accompanying research paper.

## Dashboard

An interactive Streamlit dashboard is included (`app.py`) with five modules:
1. **Buyer Segmentation Overview** — cluster distribution
2. **Financial Impact Overview** — total portfolio value, revenue contribution by segment, and revenue-share breakdown, translating cluster membership directly into financial terms
3. **Investor Behavior Dashboard** — spend and loan patterns by segment
4. **Geographic Buyer Analysis** — segment composition by country
5. **Segment Insights Panel** — descriptive statistics per segment

Users can filter by country, region, acquisition purpose, and client type.

## Tech Stack

- Python (pandas, scikit-learn, matplotlib, scipy)
- Google Colab (data processing and clustering)
- Streamlit (dashboard)

## Files

- `app.py` — Streamlit dashboard application
- `clients_final.csv` — processed client dataset with cluster labels
- `requirements.txt` — Python package dependencies
- `buyer_segmentation_parcl.ipynb` — full analysis notebook (data cleaning, feature engineering, clustering, evaluation)
- `Buyer_Segmentation_Research_Paper.pdf` — full research paper with financial impact analysis, EDA, methodology, and business recommendations

## Author

Simran
