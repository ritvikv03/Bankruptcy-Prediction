# Bankruptcy-Prediction
# Cloud-Native Bankruptcy Prediction System

A complete end-to-end machine learning pipeline built on AWS for predicting corporate bankruptcy risk. This project consolidates decentralized financial data sources into a unified data warehouse and uses machine learning to assess investment risk.

## Project Overview

Built for MSBA Financial Group to prototype a cloud-native data architecture that:
- Consolidates three disparate data sources into a centralized warehouse
- Performs automated ETL processing using AWS Glue
- Trains a machine learning model to predict bankruptcy risk
- Provides actionable investment recommendations for portfolio management

## Architecture

```
Raw Data Sources (S3)
    ↓
AWS Glue Crawlers (Schema Discovery)
    ↓
Glue Data Catalog
    ↓
Glue ETL Job (Spark)
    ↓
Processed Data (S3 Parquet)
    ↓
    ├→ Amazon Redshift (Data Warehouse)
    └→ Amazon SageMaker (Machine Learning)
         ↓
    Predictions & Recommendations
```

## Dataset

**Training Data:** 6,819 companies with 24 financial features
- **Source 1:** Balance sheet and income statement data (datacorp_financials.csv)
- **Source 2:** Financial ratios calculated by analysts (msba_fg_ratios.csv)
- **Source 3:** Historical bankruptcy records (msba_fg_bankruptcy.txt - pipe-delimited)

**Prediction Data:** 10 companies evaluated for new portfolio

**Bankruptcy Rate:** 3.23% (220 bankrupt / 6,599 healthy)

## Technologies Used

- **AWS S3** - Data lake for raw and processed data storage
- **AWS Glue** - Data cataloging and ETL processing (PySpark)
- **Amazon Redshift** - Cloud data warehouse for SQL analytics
- **Amazon SageMaker** - Machine learning model development (Jupyter notebooks)
- **Python Libraries:** pandas, scikit-learn, matplotlib, seaborn, boto3

## Implementation Steps

### 1. Data Lake Setup (S3)
Created organized bucket structure:
```
msba-financial-data-lake/
├── raw-data/
│   ├── datacorp_financial_data.csv
│   ├── msba_fg_ratio_data.csv
│   └── msba_fg_bankruptcy.txt
├── processed-data/
│   └── combined_financial_data/ (Parquet)
├── machine-learning-data/
│   └── company_profiles_to_predict_unlabeled.csv
└── predictions/
    └── bankruptcy_predictions.csv
```

### 2. Data Cataloging (Glue Crawlers)
- Created custom pipe-delimiter classifier for bankruptcy file
- Ran 3 crawlers to auto-discover schemas
- Cataloged tables in `msba_financial_db` database

### 3. ETL Processing (Glue Spark Job)
Key challenge: Bankruptcy file used different column naming (`company` vs `company_id`)

**Solution:** Used LEFT JOIN to preserve all companies
```python
# Rename bankruptcy column for consistency
bankruptcy_df = bankruptcy_df.withColumnRenamed("company", "company_id")

# LEFT JOIN to keep all companies
final_df = combined_df.join(bankruptcy_df, "company_id", "left")

# Fill missing bankruptcy flags with 0
final_df = final_df.fillna({'bankrupt': 0})
```

### 4. Data Warehouse (Redshift)
- Loaded processed Parquet data via Spectrum external schema
- Created local `financial_data` table for fast SQL queries
- Enabled SQL-based analytics for business teams

### 5. Machine Learning (SageMaker)

**Model:** Random Forest Classifier
- 100 estimators
- Max depth: 10
- Class weight: balanced (handles 3.23% imbalance)
- Random state: 42

**Performance Metrics:**
- **Accuracy:** 95.5%
- **ROC-AUC:** 0.940
- **Precision:** 38% (bankruptcy class)
- **Recall:** 64% (catches ~2/3 of bankruptcies)

## Key Findings

### Finding 1: Financial Profile Differences
Companies that went bankrupt showed dramatically different characteristics:
- **Debt ratio:** 69% higher (0.187 vs 0.111)
- **Net income to total assets:** 9% lower (0.738 vs 0.810)
- **ROA after tax:** 19% worse (0.457 vs 0.562)

### Finding 2: Top Predictive Indicators

**Risk Indicators (Positive Correlation):**
1. Debt ratio percentage (+0.25)
2. Borrowing dependency (+0.18)
3. Current liability to current assets (+0.17)

**Protective Factors (Negative Correlation):**
1. Net income to total assets (-0.32)
2. ROA before interest after tax (-0.28)
3. Net worth to assets (-0.25)

**Most Important Features (Model):**
1. Persistent EPS (0.120)
2. Borrowing dependency (0.095)
3. Net income to total assets (0.092)

## Investment Recommendations

### Portfolio Evaluation Results

**HIGH RISK - AVOID (2 companies):**
- Western Corp: 92.8% bankruptcy risk
- Innocore: 68.7% bankruptcy risk

**LOW RISK - RECOMMEND (8 companies):**
- Design Solutions: 4.7%
- Hallandall AG: 2.6%
- Highwood & Hart: 2.5%
- Pharmasolve: 0.0%
- Songster Inc: 0.0%
- Ninetech: 0.0%
- Rogers and Sons: 0.0%
- Foster & Kruse: 0.0%

**Final Recommendation:** Invest in 8 companies, avoid 2 high-risk companies

## Repository Structure

```
├── Bankruptcy_Pred.pdf              # Complete Jupyter notebook (PDF export)
├── msba-project-ritvik.ipynb        # Original notebook file
├── Gen_Bus_780_Final_Project.pdf    # Assignment requirements
├── images/                          # Visualizations and screenshots
│   ├── Bankruptcy_Pred.png
│   ├── CM_ROC-AUC.png
│   ├── Feature_Variable_Corr_Comparison.png
│   ├── Financial_Ratios_Bankrupt_Comparison.png
│   ├── Financial_Ratios_Dist_Comparison.png
│   └── Top_Feature_Bankruptcy_Pred.png
└── README.md                        # This file
```

## Key Learnings

1. **Data Integration Challenges:** Different file formats (CSV vs pipe-delimited TXT) and inconsistent column naming required careful ETL design
2. **Join Strategy Impact:** Using INNER JOIN initially excluded companies not in all source files; switching to LEFT JOIN preserved data completeness
3. **Class Imbalance:** With only 3.23% bankruptcy rate, used balanced class weights and focused on recall to catch actual bankruptcies
4. **Business Context:** ML model provides risk assessment, but final investment decisions should consider additional qualitative factors

## Academic Context

**Course:** Gen Bus 780 - Cloud Computing & Data Analytics  
**Institution:** University of Wisconsin - Madison  
**Semester:** Fall 2025  


## Notes

- Model predictions are based on historical patterns and current financial ratios
- Design Solutions (4.7% risk) was not in historical training data; prediction based purely on current financial health indicators
- Risk thresholds: <20% = LOW RISK, 20-50% = MEDIUM RISK, >50% = HIGH RISK
- All AWS resources configured with LabRole for student account compatibility

## Acknowledgments

Assignment designed by University of Wisconsin-Madison School of Business faculty. Data sources simulated for educational purposes.

---

**Author:** Ritvik  
**Date:** September 2025  
**AWS Services:** S3, Glue, Redshift, SageMaker  
**Python Version:** 3.10
