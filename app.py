# app.py
# Compact single-page FTD Survival App
# Run: streamlit run app.py

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter

# -----------------------------
# Dark theme + compact plots
# -----------------------------
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (5.2, 2.6),
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_DATA_PATH = "cell_data.csv"

FEATURE_COLS = [
    "neurite_growth",
    "calcium_rate",
    "aggregation_slope",
    "microglia_contact_time",
]

DURATION_COL = "time_to_failure_hours"
EVENT_COL = "event_observed"

PENALIZER = 0.05


# -----------------------------
# Helpers
# -----------------------------
def load_dataframe(uploaded_file, fallback_path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if not os.path.exists(fallback_path):
        st.error("Upload a CSV or place cell_data.csv next to app.py.")
        st.stop()

    return pd.read_csv(fallback_path)


def validate_columns(df):
    required = FEATURE_COLS + [DURATION_COL, EVENT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()


def verbal_prediction(std_features, summary):
    coefs = summary["coef"].to_dict()
    contribs = {k: coefs[k] * std_features[k] for k in FEATURE_COLS}
    lin = sum(contribs.values())
    risk = math.exp(lin)

    if risk > 2:
        bucket = "HIGH"
    elif risk > 1.2:
        bucket = "MODERATE"
    else:
        bucket = "LOW"

    top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    text = f"""
**Relative risk:** {risk:.2f}× baseline  
**Risk level:** {bucket}

Top drivers:
- {top[0][0]}
- {top[1][0]}
- {top[2][0]}
"""
    return text


def simple_chat(msg, summary):
    msg = msg.lower()

    if "important" in msg:
        s = summary.copy()
        s["abs"] = s["coef"].abs()
        top = s.sort_values("abs", ascending=False).head(3)
        return f"Strongest predictors: {', '.join(top.index)}"

    if "hazard" in msg:
        return "Hazard ratio >1 means faster failure. <1 means protective."

    return "Ask about feature importance or hazard ratios."


# -----------------------------
# APP
# -----------------------------
st.set_page_config(layout="centered")
st.title("FTD Cell Survival Model")

# Sidebar upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_dataframe(uploaded_file, DEFAULT_DATA_PATH)
validate_columns(df)

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

summary = cph.summary.loc[FEATURE_COLS, ["coef", "exp(coef)", "p"]]

# =========================
# 1) Predict
# =========================
st.header("Predict")

col1, col2 = st.columns(2)

with col1:
    st.dataframe(summary, height=200)

with col2:
    inputs = {}
    for f in FEATURE_COLS:
        inputs[f] = st.number_input(f, value=float(df[f].mean()))

    if st.button("Predict"):
        arr = np.array([[inputs[f] for f in FEATURE_COLS]])
        arr_s = scaler.transform(arr)[0]
        std = {FEATURE_COLS[i]: arr_s[i] for i in range(len(FEATURE_COLS))}
        st.markdown(verbal_prediction(std, summary))

st.divider()

# =========================
# 2) Visualize
# =========================
st.header("Visualize")

graph_choice = st.selectbox("Select Graph", [
    "Feature Histogram",
    "Correlation Heatmap",
    "Kaplan–Meier Overall",
    "Kaplan–Meier Split",
    "Hazard Ratios",
    "Risk Distribution"
])

# Histogram
if graph_choice == "Feature Histogram":
    feat = st.selectbox("Feature", FEATURE_COLS)
    fig, ax = plt.subplots()
    ax.hist(df[feat], bins=30)
    ax.set_title(feat)
    fig.tight_layout()
    st.pyplot(fig)

# Correlation
elif graph_choice == "Correlation Heatmap":
    corr = df[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
    ax.set_yticklabels(FEATURE_COLS)
    fig.colorbar(im)
    fig.tight_layout()
    st.pyplot(fig)

# KM Overall
elif graph_choice == "Kaplan–Meier Overall":
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()
    kmf.fit(df[DURATION_COL], df[EVENT_COL])
    kmf.plot(ax=ax)
    ax.set_title("Overall Survival")
    fig.tight_layout()
    st.pyplot(fig)

# KM Split
elif graph_choice == "Kaplan–Meier Split":
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
    ax.set_title(f"Survival by {feat}")
    fig.tight_layout()
    st.pyplot(fig)

# Hazard Ratios
elif graph_choice == "Hazard Ratios":
    hr = summary["exp(coef)"]
    fig, ax = plt.subplots()
    ax.bar(hr.index, hr.values)
    ax.axhline(1.0)
    ax.set_yscale("log")
    ax.set_title("Hazard Ratios")
    fig.tight_layout()
    st.pyplot(fig)

# Risk Distribution
elif graph_choice == "Risk Distribution":
    risk = cph.predict_partial_hazard(df_scaled[FEATURE_COLS])
    fig, ax = plt.subplots()
    ax.hist(risk, bins=40)
    ax.set_title("Predicted Risk Distribution")
    fig.tight_layout()
    st.pyplot(fig)

st.divider()

# =========================
# 3) Chat
# =========================
st.header("Chat")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

user_msg = st.chat_input("Ask about hazard ratios or features...")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    reply = simple_chat(user_msg, summary)
    st.session_state.chat.append(("assistant", reply))

    with st.chat_message("assistant"):
        st.markdown(reply)