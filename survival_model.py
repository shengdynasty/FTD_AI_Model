import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "cell_data.csv"

GROUP_COL = "donor_id"
DURATION_COL = "time_to_failure_hours"
EVENT_COL = "event_observed"

FEATURE_COLS = [
    "neurite_growth",
    "calcium_rate",
    "aggregation_slope",
    "microglia_contact_time"
]

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(CSV_PATH)

required = FEATURE_COLS + [GROUP_COL, DURATION_COL, EVENT_COL]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()
df[EVENT_COL] = df[EVENT_COL].astype(int)
df[GROUP_COL] = df[GROUP_COL].astype(str)

print("Rows:", len(df))
print("Groups:", df[GROUP_COL].nunique())
print("Event rate:", df[EVENT_COL].mean())

# -----------------------------
# GROUP CV
# -----------------------------
groups = df[GROUP_COL].values
gkf = GroupKFold(n_splits=min(5, df[GROUP_COL].nunique()))

c_indices = []

for fold, (tr, te) in enumerate(gkf.split(df, groups=groups), start=1):
    train = df.iloc[tr].copy()
    test = df.iloc[te].copy()

    # Scale features (fit on train only!)
    scaler = StandardScaler()
    train[FEATURE_COLS] = scaler.fit_transform(train[FEATURE_COLS])
    test[FEATURE_COLS] = scaler.transform(test[FEATURE_COLS])

    # lifelines CoxPH expects a single dataframe including duration+event
    cph = CoxPHFitter(penalizer=0.05)  # penalizer helps stability
    cph.fit(train[FEATURE_COLS + [DURATION_COL, EVENT_COL]],
            duration_col=DURATION_COL,
            event_col=EVENT_COL)

    # Predict partial hazards on test
    partial_hazards = cph.predict_partial_hazard(test[FEATURE_COLS])

    # Concordance index (higher is better, 0.5 random)
    c_index = concordance_index(
        test[DURATION_COL],
        -partial_hazards.values.ravel(),  # negative because higher hazard -> shorter time
        test[EVENT_COL]
    )
    c_indices.append(c_index)

    print(f"Fold {fold}: C-index = {c_index:.3f}")

print("\nMean C-index:", float(np.mean(c_indices)))

# -----------------------------
# TRAIN FINAL MODEL + SHOW COEFFICIENTS
# -----------------------------
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = scaler.fit_transform(df_scaled[FEATURE_COLS])

final_cph = CoxPHFitter(penalizer=0.05)
final_cph.fit(df_scaled[FEATURE_COLS + [DURATION_COL, EVENT_COL]],
              duration_col=DURATION_COL,
              event_col=EVENT_COL)

print("\nFinal Cox model coefficients (positive => faster failure):")
print(final_cph.summary[["coef", "exp(coef)", "p"]])