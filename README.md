# Elevate Labs — Data Science Internship

This repository contains all **6 weekly tasks** completed during the **Elevate Labs Data Science Internship**. Each week covers a core machine learning concept, applied to a real-world dataset.

## 📂 Repository Structure

```
Elevate--Labs/
├── README.md                          ← You are here (common overview)
├── Task-1-Data-Preprocessing/
│   ├── data_preprocessing.py
│   ├── Titanic-Dataset.csv
│   └── README.md
├── Task-2-Exploratory-Data-Analysis/
│   ├── eda_titanic.py
│   ├── Titanic-Dataset.csv
│   ├── screenshots/
│   └── README.md
├── Task-3-Linear-Regression/
│   ├── linear_regression.py
│   ├── Titanic-Dataset.csv
│   ├── screenshots/
│   └── README.md
├── Task-4-Logistic-Regression/
│   ├── logistic_regression.py
│   ├── cancer-data.csv
│   ├── screenshots/
│   └── README.md
├── Task-5-Decision-Tree-Random-Forest/
│   ├── decision_tree_random_forest.py
│   ├── heart.csv
│   ├── screenshots/
│   └── README.md
└── Task-6-KNN-Classification/
    ├── knn_classification.py
    ├── Iris.csv
    ├── screenshots/
    └── README.md
```

## 📅 Weekly Task Overview

| Week | Task | Technique | Dataset | Key Learnings | Status |
|------|------|-----------|---------|---------------|--------|
| 1 | [Data Preprocessing](Task-1-Data-Preprocessing/) | Cleaning, encoding, scaling | Titanic | Missing values, one-hot encoding, StandardScaler, IQR outlier removal | ✅ Done |
| 2 | [Exploratory Data Analysis](Task-2-Exploratory-Data-Analysis/) | Visualization & statistics | Titanic | Histograms, boxplots, pairplots, correlation heatmap, group-by inferences | ✅ Done |
| 3 | [Linear Regression](Task-3-Linear-Regression/) | Regression | Titanic (Fare) | Train/test split, MAE / MSE / R², coefficient interpretation | ✅ Done |
| 4 | [Logistic Regression](Task-4-Logistic-Regression/) | Binary classification | Breast Cancer | Confusion matrix, precision/recall/F1, ROC-AUC, threshold tuning | ✅ Done |
| 5 | [Decision Tree & Random Forest](Task-5-Decision-Tree-Random-Forest/) | Ensemble methods | Heart Disease | Overfitting control, feature importance, 5-fold cross-validation | ✅ Done |
| 6 | [KNN Classification](Task-6-KNN-Classification/) | Instance-based learning | Iris | Choosing K, confusion matrix, decision boundary visualization | ✅ Done |

## ✅ What Was Done (Summary)

- **Week 1** — Took a raw dataset end-to-end through a full preprocessing pipeline: missing-value imputation, categorical encoding, feature scaling, and outlier removal.
- **Week 2** — Performed EDA on the Titanic dataset: distribution plots, pairplot, correlation heatmap, and survival-rate inferences by sex, class, and age group.
- **Week 3** — Built a linear regression model to predict ticket fare, evaluated it with MAE/MSE/R², and visualized actual vs predicted values.
- **Week 4** — Built a logistic regression classifier on the breast cancer dataset, evaluated with ROC-AUC and precision-recall, and experimented with threshold tuning (0.5 vs 0.7).
- **Week 5** — Trained decision tree and random forest classifiers on heart disease data, diagnosed and controlled overfitting, ranked feature importances, and validated with cross-validation.
- **Week 6** — Implemented KNN on the Iris dataset, experimented with K values (1–15), picked the best K, and visualized the confusion matrix and decision boundaries.

## 🚀 How to Run Any Task

```bash
cd Task-X-<Task-Name>
python <script_name>.py
```

All scripts read their CSV from the same folder, so run them from inside the task folder.

## 🛠️ Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

Install everything with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 📌 Next Steps (What's Pending / What I Need)

- [ ] **Model persistence:** save trained models with `joblib`/`pickle` for reuse
- [ ] **Hyperparameter tuning:** add GridSearchCV for Task 4, 5, and 6 models
- [ ] **New datasets:** apply the same pipelines to a fresh dataset (e.g., loan approval or churn prediction)
- [ ] **Deployment:** wrap one end-to-end pipeline in a simple Streamlit/Flask app
- [ ] **Git hygiene:** initialize a git repository, add `.gitignore` (for `__pycache__/`), and push to GitHub

> Note: all 6 assigned weekly tasks are **complete** — nothing is overdue. The items above are optional improvements to strengthen the portfolio.
