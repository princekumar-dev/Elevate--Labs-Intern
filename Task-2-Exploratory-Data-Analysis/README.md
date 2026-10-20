# Week 2 — Exploratory Data Analysis (EDA)

## Objective
Explore and visualize a dataset to find patterns, trends, and relationships between features.

## Dataset
`Titanic-Dataset.csv` — Titanic passenger data.

## What Was Done
- Loaded the dataset and printed `df.info()`, summary statistics (`describe(include='all')`), and per-column missing values
- Plotted a **histogram + boxplot** for every numeric feature to study distribution and outliers
- Built a **pairplot** (sampled to 500 rows) of numeric features, colored by survival
- Computed and plotted the **correlation matrix** as an annotated heatmap
- Derived simple inferences:
  - Overall survival rate
  - Survival rate by **sex**, **passenger class**, and **age group** (child / teen / young adult / adult / senior)

## How to Run
```bash
python eda_titanic.py
```

## Requirements
`pandas`, `numpy`, `matplotlib`, `seaborn`

## Outputs
All plots generated are saved in the [`screenshots/`](screenshots/) folder.
