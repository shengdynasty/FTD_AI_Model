# FTD Cell-Level Failure Prediction Model


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

import xgboost as xgb
import shap
import matplotlib.pyplot as plt


# config model thingy

CSV_PATH = "cell_data.csv"   # datafile replace here
LABEL_COL = "failed_7d"      # 7 day fail
GROUP_COL = "donor_id"       # split donor

FEATURE_COLS = [
    "neurite_growth",
    "calcium_rate",
    "aggregation_slope",
    "microglia_contact_time"
]



#load data
df = pd.read_csv(CSV_PATH)

# check stuff
required_cols = FEATURE_COLS + [LABEL_COL, GROUP_COL]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df.dropna(subset=required_cols)

X = df[FEATURE_COLS]
y = df[LABEL_COL].astype(int)
groups = df[GROUP_COL]

print("Total rows:", len(df))
print("Number of donors:", groups.nunique())
print("Failure rate:", y.mean())



# GROUP CROSS VALIDATION
gkf = GroupKFold(n_splits=min(5, groups.nunique()))

auc_scores = []
pr_scores = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    auc_scores.append(auc)
    pr_scores.append(pr)

    print(f"\n--- Fold {fold} ---")
    print("AUROC:", round(auc, 3))
    print("PR-AUC:", round(pr, 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))


print("\n==============================")
print("Mean AUROC:", round(np.mean(auc_scores), 3))
print("Mean PR-AUC:", round(np.mean(pr_scores), 3))
print("==============================")



# TRAIN FINAL MODEL ON FULL DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    eval_metric="logloss"
)

final_model.fit(X_scaled, y)

print("\nFinal model trained.")



# SHAP INTERPRETATION

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_scaled)

print("\nGenerating SHAP plots...")

# Global importance
shap.summary_plot(shap_values, X_scaled, feature_names=FEATURE_COLS, plot_type="bar")

# Detailed impact plot
shap.summary_plot(shap_values, X_scaled, feature_names=FEATURE_COLS)

print("Done.")