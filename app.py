# app.py
# Compact single-page FTD Survival App
# - Upload + Chat (chat bubbles) in sidebar
# - Predict in main
# - Visualize with tabs
#
# Run:
#   streamlit run app.py

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
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")
            st.stop()

    if not os.path.exists(fallback_path):
        st.sidebar.error("Upload a CSV or place cell_data.csv next to app.py.")
        st.stop()

    try:
        return pd.read_csv(fallback_path)
    except Exception as e:
        st.sidebar.error(f"Failed to read {fallback_path}: {e}")
        st.stop()


def validate_columns(df):
    required = FEATURE_COLS + [DURATION_COL, EVENT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.sidebar.error(f"Missing required columns: {missing}")
        st.stop()


def verbal_prediction(std_features, summary):
    coefs = summary["coef"].to_dict()
    contribs = {k: coefs[k] * float(std_features[k]) for k in FEATURE_COLS}
    lin = sum(contribs.values())
    risk = math.exp(lin)

    if risk > 2:
        bucket = "HIGH"
    elif risk > 1.2:
        bucket = "MODERATE"
    else:
        bucket = "LOW"

    top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    top_lines = "\n".join([f"- **{feat}** ({'↑' if val > 0 else '↓'})" for feat, val in top])

    return f"""
**Relative risk:** `{risk:.2f}×` baseline  
**Risk level:** **{bucket}**

**Top drivers:**
{top_lines}

**Rule of thumb:** HR>1 increases risk, HR<1 is protective.
"""


def simple_chat(msg, summary):
    msg = msg.lower().strip()

    if any(k in msg for k in ["most important", "strongest", "top", "largest effect"]):
        s = summary.copy()
        s["abs"] = s["coef"].abs()
        top = s.sort_values("abs", ascending=False).head(3)
        return f"Strongest predictors: {', '.join(top.index)}."

    if "microglia" in msg:
        hr = float(summary.loc["microglia_contact_time", "exp(coef)"])
        return f"microglia_contact_time HR={hr:.2f}: more contact → faster failure."

    if "aggregation" in msg or "tau" in msg or "tdp" in msg:
        hr = float(summary.loc["aggregation_slope", "exp(coef)"])
        return f"aggregation_slope HR={hr:.2f}: higher early aggregation → faster failure."

    if "calcium" in msg:
        hr = float(summary.loc["calcium_rate", "exp(coef)"])
        return f"calcium_rate HR={hr:.2f}: HR<1 means higher calcium activity is protective."

    if "neurite" in msg:
        hr = float(summary.loc["neurite_growth", "exp(coef)"])
        return f"neurite_growth HR={hr:.2f}: mild protective effect if HR<1."

    if "hazard" in msg or "hr" in msg:
        return "Hazard ratio (HR) is relative risk over time. HR>1 faster failure; HR<1 slower failure."

    return "Ask: 'most important feature?', 'explain microglia', or 'what is hazard ratio?'."


def render_sidebar_chat():
    """
    Sidebar chat UI using a styled markdown bubble list + input box.
    (Streamlit's st.chat_message is not supported inside st.sidebar reliably,
     so we mimic chat bubbles with HTML/CSS in markdown.)
    """
    st.sidebar.subheader("Chat")

    # Minimal CSS for bubbles in sidebar
    st.sidebar.markdown(
        """
        <style>
        .chat-wrap { display: flex; flex-direction: column; gap: 8px; }
        .bubble {
            padding: 8px 10px;
            border-radius: 10px;
            line-height: 1.25;
            font-size: 0.88rem;
            word-wrap: break-word;
        }
        .user {
            background: rgba(80, 160, 255, 0.25);
            border: 1px solid rgba(80, 160, 255, 0.35);
            align-self: flex-end;
        }
        .assistant {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            align-self: flex-start;
        }
        .meta { opacity: 0.7; font-size: 0.75rem; margin-bottom: 2px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Show last N messages to keep sidebar compact
    last_n = 10
    msgs = st.session_state.chat[-last_n:]

    # Render bubbles
    html = ['<div class="chat-wrap">']
    for role, content in msgs:
        label = "You" if role == "user" else "Assistant"
        klass = "user" if role == "user" else "assistant"
        safe = (
            str(content)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        html.append(f'<div><div class="meta">{label}</div><div class="bubble {klass}">{safe}</div></div>')
    html.append("</div>")

    st.sidebar.markdown("".join(html), unsafe_allow_html=True)

    # Input row
    user_msg = st.sidebar.text_input("Message", key="sidebar_chat_input", placeholder="Ask about HR, features...")
    col_send, col_clear = st.sidebar.columns([1, 1])
    send = col_send.button("Send", key="sidebar_chat_send")
    clear = col_clear.button("Clear", key="sidebar_chat_clear")

    return user_msg, send, clear


# -----------------------------
# APP
# -----------------------------
st.set_page_config(layout="centered")
st.title("FTD Cell Survival Model")

# ==============
# Sidebar: Upload
# ==============
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_dataframe(uploaded_file, DEFAULT_DATA_PATH)
validate_columns(df)

df = df.dropna(subset=FEATURE_COLS + [DURATION_COL, EVENT_COL]).copy()
df[EVENT_COL] = df[EVENT_COL].astype(int)

st.sidebar.caption(f"Rows: {len(df)} | Event rate: {df[EVENT_COL].mean():.2f}")

# Fit model (so chat can reference it too)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = scaler.fit_transform(df_scaled[FEATURE_COLS])

cph = CoxPHFitter(penalizer=PENALIZER)
cph.fit(
    df_scaled[FEATURE_COLS + [DURATION_COL, EVENT_COL]],
    duration_col=DURATION_COL,
    event_col=EVENT_COL
)

summary = cph.summary.loc[FEATURE_COLS, ["coef", "exp(coef)", "p"]]

st.sidebar.divider()

# ==============
# Sidebar: Chat bubbles
# ==============
user_msg, send, clear = render_sidebar_chat()

if clear:
    st.session_state.chat = []
    st.rerun()

if send and user_msg.strip():
    st.session_state.chat.append(("user", user_msg.strip()))
    reply = simple_chat(user_msg, summary)
    st.session_state.chat.append(("assistant", reply))
    # clear the input field
    st.session_state["sidebar_chat_input"] = ""
    st.rerun()

# ==============
# Main: Predict
# ==============
st.header("Predict")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cox Summary")
    st.dataframe(summary, height=220, use_container_width=True)

with col2:
    st.subheader("Single-cell verbal prediction")
    inputs = {f: st.number_input(f, value=float(df[f].mean())) for f in FEATURE_COLS}

    if st.button("Predict"):
        arr = np.array([[inputs[f] for f in FEATURE_COLS]])
        arr_s = scaler.transform(arr)[0]
        std = {FEATURE_COLS[i]: float(arr_s[i]) for i in range(len(FEATURE_COLS))}
        st.markdown(verbal_prediction(std, summary))

st.divider()

# ==============
# Main: Visualize (TABS)
# ==============
st.header("Visualize")

tab_hist, tab_corr, tab_km, tab_km_split, tab_hr, tab_risk = st.tabs([
    "Histogram",
    "Correlation",
    "KM Overall",
    "KM Split",
    "Hazard Ratios",
    "Risk Dist."
])

with tab_hist:
    feat = st.selectbox("Feature", FEATURE_COLS, key="hist_feat")
    bins = st.slider("Bins", 10, 80, 30, 5, key="hist_bins")

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.hist(df[feat].dropna(), bins=bins)
    ax.set_title(f"Histogram: {feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)

with tab_corr:
    corr = df[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
    ax.set_yticklabels(FEATURE_COLS)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)

with tab_km:
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    kmf.fit(df[DURATION_COL], df[EVENT_COL], label="All")
    kmf.plot(ax=ax, ci_show=True)
    ax.set_title("Kaplan–Meier (Overall)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Survival")
    fig.tight_layout()
    st.pyplot(fig)

with tab_km_split:
    feat = st.selectbox(
        "Split Feature",
        FEATURE_COLS,
        index=FEATURE_COLS.index("microglia_contact_time"),
        key="km_split_feat"
    )
    med = df[feat].median()
    high = df[df[feat] >= med]
    low = df[df[feat] < med]

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(5.8, 3.1))

    kmf.fit(low[DURATION_COL], low[EVENT_COL], label="Low")
    kmf.plot(ax=ax, ci_show=True)
    kmf.fit(high[DURATION_COL], high[EVENT_COL], label="High")
    kmf.plot(ax=ax, ci_show=True)

    ax.set_title(f"KM Split by {feat}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Survival")
    fig.tight_layout()
    st.pyplot(fig)

with tab_hr:
    hr = summary["exp(coef)"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(hr.index, hr.values)
    ax.axhline(1.0)
    ax.set_yscale("log")
    ax.set_title("Hazard Ratios (log)")
    ax.set_ylabel("HR")
    ax.set_xticklabels(hr.index, rotation=25, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

with tab_risk:
    risk = cph.predict_partial_hazard(df_scaled[FEATURE_COLS]).values.ravel()
    bins = st.slider("Bins", 10, 120, 40, 5, key="risk_bins")

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.hist(risk, bins=bins)
    ax.set_title("Predicted Risk (Partial Hazard)")
    ax.set_xlabel("Partial hazard")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)

st.caption("Note: Cox predicts relative hazard over time. Exact time prediction needs calibration.")