# app.py
# Minimal Research Dashboard Version
# Run: streamlit run app.py

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter

# -----------------------
# Compact dark plots
# -----------------------
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (4.8, 2.4),
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "cell_data.csv"

FEATURE_COLS = [
    "neurite_growth",
    "calcium_rate",
    "aggregation_slope",
    "microglia_contact_time",
]

DURATION_COL = "time_to_failure_hours"
EVENT_COL = "event_observed"

PENALIZER = 0.05


# -----------------------
# Helpers
# -----------------------
def load_data(upload):
    if upload:
        return pd.read_csv(upload)
    if not os.path.exists(DATA_PATH):
        st.error("Upload CSV or include cell_data.csv.")
        st.stop()
    return pd.read_csv(DATA_PATH)


def validate(df):
    required = FEATURE_COLS + [DURATION_COL, EVENT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()


def verbal_pred(std, summary):
    coefs = summary["coef"].to_dict()
    contrib = {k: coefs[k] * std[k] for k in FEATURE_COLS}
    risk = math.exp(sum(contrib.values()))

    if risk > 2:
        lvl = "HIGH"
    elif risk > 1.2:
        lvl = "MODERATE"
    else:
        lvl = "LOW"

    return f"Risk: {risk:.2f}× | Level: {lvl}"


# -----------------------
# UI
# -----------------------
st.set_page_config(layout="wide")
st.title("FTD Survival Dashboard")

upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(upload)
validate(df)

df = df.dropna(subset=FEATURE_COLS + [DURATION_COL, EVENT_COL])
df[EVENT_COL] = df[EVENT_COL].astype(int)

# Fit model
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = scaler.fit_transform(df_scaled[FEATURE_COLS])

cph = CoxPHFitter(penalizer=PENALIZER)
cph.fit(df_scaled[FEATURE_COLS + [DURATION_COL, EVENT_COL]],
        duration_col=DURATION_COL,
        event_col=EVENT_COL)

summary = cph.summary.loc[FEATURE_COLS, ["coef", "exp(coef)"]]

# -----------------------
# Layout (2-column dashboard)
# -----------------------
left, right = st.columns([1, 1.4])

# ===== LEFT PANEL =====
with left:
    st.subheader("Predict")

    inputs = {}
    for f in FEATURE_COLS:
        inputs[f] = st.number_input(f, value=float(df[f].mean()))

    if st.button("Run"):
        arr = np.array([[inputs[f] for f in FEATURE_COLS]])
        arr_s = scaler.transform(arr)[0]
        std = {FEATURE_COLS[i]: arr_s[i] for i in range(len(FEATURE_COLS))}
        st.success(verbal_pred(std, summary))

    st.divider()

    st.subheader("Model Coefficients")
    st.dataframe(summary, height=180)

    st.divider()

    st.subheader("Chat")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask about hazard ratios")
    if msg:
        st.session_state.chat.append(msg)

    for m in st.session_state.chat[-3:]:
        st.caption(f"> {m}")

# ===== RIGHT PANEL =====
with right:
    st.subheader("Visualization")

    graph = st.selectbox("Graph Type", [
        "Histogram",
        "Correlation",
        "KM Overall",
        "KM Split",
        "Hazard Ratios",
        "Risk Distribution"
    ])

    if graph == "Histogram":
        feat = st.selectbox("Feature", FEATURE_COLS)
        fig, ax = plt.subplots()
        ax.hist(df[feat], bins=30)
        ax.set_title(feat)
        fig.tight_layout()
        st.pyplot(fig)

    elif graph == "Correlation":
        corr = df[FEATURE_COLS].corr()
        fig, ax = plt.subplots(figsize=(4.8, 3))
        im = ax.imshow(corr.values, vmin=-1, vmax=1)
        ax.set_xticks(range(len(FEATURE_COLS)))
        ax.set_yticks(range(len(FEATURE_COLS)))
        ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
        ax.set_yticklabels(FEATURE_COLS)
        fig.colorbar(im)
        fig.tight_layout()
        st.pyplot(fig)

    elif graph == "KM Overall":
        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots()
        kmf.fit(df[DURATION_COL], df[EVENT_COL])
        kmf.plot(ax=ax)
        ax.set_title("Overall Survival")
        fig.tight_layout()
        st.pyplot(fig)

    elif graph == "KM Split":
        feat = st.selectbox("Split Feature", FEATURE_COLS)
        med = df[feat].median()
        high = df[df[feat] >= med]
        low = df[df[feat] < med]

        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots()
        kmf.fit(low[DURATION_COL], low[EVENT_COL], label="Low")
        kmf.plot(ax=ax)
        kmf.fit(high[DURATION_COL], high[EVENT_COL], label="High")
        kmf.plot(ax=ax)
        ax.set_title(feat)
        fig.tight_layout()
        st.pyplot(fig)

    elif graph == "Hazard Ratios":
        hr = summary["exp(coef)"]
        fig, ax = plt.subplots()
        ax.bar(hr.index, hr.values)
        ax.axhline(1.0)
        ax.set_yscale("log")
        ax.set_title("Hazard Ratios")
        fig.tight_layout()
        st.pyplot(fig)

    elif graph == "Risk Distribution":
        risk = cph.predict_partial_hazard(df_scaled[FEATURE_COLS])
        fig, ax = plt.subplots()
        ax.hist(risk, bins=40)
        ax.set_title("Risk Distribution")
        fig.tight_layout()
        st.pyplot(fig)