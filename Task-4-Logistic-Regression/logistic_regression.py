import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve
)

# ==========================================
# 1. Load the Breast Cancer Dataset (local CSV)
# ==========================================
# Load CSV (expects `cancer-data.csv` in the same folder)
df = pd.read_csv('cancer-data.csv')

# Drop any unnamed/all-NaN columns (sometimes created by trailing commas)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Drop identifier column if present
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Map diagnosis to binary target: Malignant (M) -> 1, Benign (B) -> 0
if 'diagnosis' in df.columns:
    df['target'] = (df['diagnosis'].astype(str).str.upper() == 'M').astype(int)
else:
    raise ValueError('Expected column "diagnosis" not found in cancer-data.csv')

# Use all numeric feature columns (drop diagnosis and target)
X = df.drop(columns=['diagnosis', 'target'], errors='ignore')
y = df['target']

# ==========================================
# 2. Train/Test Split and Standardize Features
# ==========================================
# 80/20 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization (Crucial for Logistic Regression regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. Fit a Logistic Regression Model
# ==========================================
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Get predicted probabilities (needed for ROC-AUC and threshold tuning)
# predict_proba returns [prob_class_0, prob_class_1]
y_probs = model.predict_proba(X_test_scaled)[:, 1] 

# Default predictions (using 0.5 threshold)
y_pred_default = model.predict(X_test_scaled)

# ==========================================
# 4. Evaluate the Model
# ==========================================
print("--- Evaluation with Default Threshold (0.5) ---")
print("\ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_default))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_default))

roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")

# ==========================================
# 5. Tune Threshold & Plotting
# ==========================================
# Let's say we want to optimize for Precision or Recall by shifting the threshold.
# Example: Setting a strict threshold of 0.7 instead of 0.5
custom_threshold = 0.7
y_pred_custom = (y_probs >= custom_threshold).astype(int)

print(f"--- Evaluation with Custom Threshold ({custom_threshold}) ---")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))

# --- PLOTTING ROC AND PRECISION-RECALL CURVES ---
fpr, tpr, _ = roc_curve(y_test, y_probs)
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(12, 5))

# Plot ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Plot Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()