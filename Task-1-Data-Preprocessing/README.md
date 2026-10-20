# Week 1 — Data Preprocessing

## Objective
Clean and prepare a raw dataset so it is ready for machine learning models.

## Dataset
`Titanic-Dataset.csv` — Titanic passenger data.

## What Was Done
- Loaded the dataset and inspected structure (`df.info()`), missing values, and summary statistics
- **Missing values:** filled numeric columns with the *mean* and categorical columns with the *mode*
- **Encoding:** converted categorical features to numeric using one-hot encoding (`drop_first=True`)
- **Scaling:** standardized all numeric features with `StandardScaler`
- **Outlier detection:** visualized outliers with boxplots, then removed them using the **IQR method** (1.5 × IQR rule)
- Verified the final cleaned dataset shape

## How to Run
```bash
python data_preprocessing.py
```

## Requirements
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
