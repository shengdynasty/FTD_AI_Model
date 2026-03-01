# app.py
# Single-page Streamlit app: Upload CSV -> Cox survival model -> Predict + Visualize (dropdown) + Chat
#
# Run:
#   streamlit run app.py
#
# Expected CSV columns (survival mode):
#   neurite_growth, calcium_rate, aggregation_slope, microglia_contact_time,
#   time_to_failure_hours, event_observed
#
# Optional:
#   donor_id (not required)

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter

# -----------------------------
# Dark theme for all matplotlib plots
# -----------------------------
plt.style.use("dark_background")


# =========================
# CONFIG
# =========================
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
MAX_PREVIEW_ROWS = 50


# =========================
# HELPERS
# =========================
def load_dataframe(uploaded_file, fallback_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()

    if not os.path.exists(fallback_path):
        st.error(
            f"Could not find `{fallback_path}` and no CSV was uploaded.\n\n"
            "Upload a CSV using the sidebar, or place `cell_data.csv` next to app.py."
        )
        st.stop()

    try:
        return pd.read_csv(fallback_path)
    except Exception as e:
        st.error(f"Failed to read `{fallback_path}`: {e}")
        st.stop()


def validate_columns(df: pd.DataFrame) -> None:
    required = FEATURE_COLS + [DURATION_COL, EVENT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "CSV is missing required columns:\n\n"
            + "\n".join([f"- {c}" for c in missing])
            + "\n\nExpected at minimum:\n"
            + "\n".join([f"- {c}" for c in required])
        )
        st.stop()


def describe_effects(cox_summary: pd.DataFrame) -> str:
    lines = []
    for feat, row in cox_summary.iterrows():
        coef = float(row["coef"])
        hr = float(row["exp(coef)"])
        p = float(row["p"])
        direction = "increases" if coef > 0 else "decreases"
        lines.append(
            f"- **{feat}**: coef={coef:.3f}, HR={hr:.3f} (p={p:.2e}) → {direction} hazard (faster failure)."
        )
    return "\n".join(lines)


def verbal_prediction(standardized_features: dict, cox_summary: pd.DataFrame) -> tuple[str, float, str]:
    coefs = cox_summary["coef"].to_dict()
    contribs = {k: coefs[k] * float(standardized_features[k]) for k in FEATURE_COLS}
    sorted_feats = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)

    lin_pred = sum(contribs.values())
    rel_risk = math.exp(lin_pred)

    if rel_risk >= 2.0:
        bucket = "HIGH"
    elif rel_risk >= 1.2:
        bucket = "MODERATE"
    else:
        bucket = "LOW"

    top_factors = []
    for feat, val in sorted_feats[:3]:
        direction = "raises" if val > 0 else "lowers"
        top_factors.append(f"**{feat}** {direction} risk")

    md = f"""
### Verbal risk prediction
**Relative risk (hazard) estimate:** `{rel_risk:.2f}×` baseline  
**Risk bucket:** **{bucket}** (higher = likely faster failure)

**Main drivers (top 3):**
- {top_factors[0]}
- {top_factors[1]}
- {top_factors[2]}

**How to read this:**  
- Positive contributions push toward **faster failure**  
- Negative contributions push toward **slower failure**
"""
    return md, rel_risk, bucket


def simple_local_chat(user_msg: str, cox_summary: pd.DataFrame) -> str:
    msg = user_msg.lower().strip()

    if any(k in msg for k in ["most important", "strongest", "top feature", "largest effect"]):
        s = cox_summary.copy()
        s["abs_coef"] = s["coef"].abs()
        top = s.sort_values("abs_coef", ascending=False).head(3)
        items = [f"{i} (HR={top.loc[i,'exp(coef)']:.2f})" for i in top.index]
        return f"The strongest predictors (largest absolute effect) are: {', '.join(items)}."

    if "microglia" in msg:
        hr = float(cox_summary.loc["microglia_contact_time", "exp(coef)"])
        coef = float(cox_summary.loc["microglia_contact_time", "coef"])
        return (
            f"**microglia_contact_time** has HR={hr:.2f} (coef={coef:.3f}). "
            f"HR>1 means more microglia contact is associated with faster failure (higher hazard)."
        )

    if "aggregation" in msg or "tdp" in msg or "tau" in msg:
        hr = float(cox_summary.loc["aggregation_slope", "exp(coef)"])
        coef = float(cox_summary.loc["aggregation_slope", "coef"])
        return (
            f"**aggregation_slope** has HR={hr:.2f} (coef={coef:.3f}). "
            f"Higher early aggregation trend predicts faster failure."
        )

    if "calcium" in msg:
        hr = float(cox_summary.loc["calcium_rate", "exp(coef)"])
        coef = float(cox_summary.loc["calcium_rate", "coef"])
        return (
            f"**calcium_rate** has HR={hr:.2f} (coef={coef:.3f}). "
            f"Because HR<1, higher calcium activity is protective (slower failure)."
        )

    if "neurite" in msg:
        hr = float(cox_summary.loc["neurite_growth", "exp(coef)"])
        coef = float(cox_summary.loc["neurite_growth", "coef"])
        return (
            f"**neurite_growth** has HR={hr:.2f} (coef={coef:.3f}). "
            f"HR slightly below 1 suggests neurite growth is mildly protective."
        )

    if "hazard ratio" in msg or msg == "hr" or "what is hazard" in msg:
        return (
            "**Hazard ratio (HR)** is a relative risk over time.\n\n"
            "- HR > 1: higher feature values → faster failure\n"
            "- HR < 1: higher feature values → slower failure\n\n"
            "In a Cox model, effects are multiplicative on hazard."
        )

    if "p value" in msg or "p-value" in msg:
        return (
            "A **p-value** measures how strong the evidence is that a feature’s effect is not zero.\n\n"
            "Smaller p-values (e.g., <0.05) generally indicate the feature is likely associated with failure risk."
        )

    return (
        "Try asking:\n"
        "- “Which feature is most important?”\n"
        "- “Explain microglia_contact_time”\n"
        "- “What does hazard ratio mean?”\n"
        "- “Why is calcium protective?”"
    )


# =========================
# APP UI
# =========================
st.set_page_config(page_title="FTD Cell Failure Predictor", layout="wide")
st.title("FTD Cell-Level Failure Predictor")
st.caption("Single page: Predict + Visualize (dropdown) + Chat | Dark-themed plots | No external API")

# Sidebar: upload CSV
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
df = load_dataframe(uploaded_file, DEFAULT_DATA_PATH)
st.sidebar.caption("Using: " + ("Uploaded CSV" if uploaded_file is not None else DEFAULT_DATA_PATH))

# Validate required columns
validate_columns(df)

# Clean
df = df.dropna(subset=FEATURE_COLS + [DURATION_COL, EVENT_COL]).copy()
df[EVENT_COL] = df[EVENT_COL].astype(int)

# Fit scaler + Cox (demo: fit on full data)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = scaler.fit_transform(df_scaled[FEATURE_COLS])

cph = CoxPHFitter(penalizer=PENALIZER)
cph.fit(df_scaled[FEATURE_COLS + [DURATION_COL, EVENT_COL]],
        duration_col=DURATION_COL, event_col=EVENT_COL)

summary = cph.summary.loc[FEATURE_COLS, ["coef", "exp(coef)", "p"]]

# =========================
# SECTION 1: PREDICT
# =========================
st.header("1) Predict")

colA, colB = st.columns([1.1, 1.0], gap="large")

with colA:
    st.subheader("Cox Model Summary")
    st.dataframe(summary, use_container_width=True)
    st.markdown("#### Quick interpretation")
    st.markdown(describe_effects(summary))

with colB:
    st.subheader("Single-cell verbal prediction")
    st.write("Enter raw feature values (the app standardizes internally).")

    raw_inputs = {
        "neurite_growth": st.number_input("neurite_growth", value=float(df["neurite_growth"].mean())),
        "calcium_rate": st.number_input("calcium_rate", value=float(df["calcium_rate"].mean())),
        "aggregation_slope": st.number_input("aggregation_slope", value=float(df["aggregation_slope"].mean())),
        "microglia_contact_time": st.number_input("microglia_contact_time", value=float(df["microglia_contact_time"].mean())),
    }

    if st.button("Predict risk (verbal)"):
        arr = np.array([[raw_inputs[c] for c in FEATURE_COLS]])
        arr_s = scaler.transform(arr)[0]
        standardized = {FEATURE_COLS[i]: float(arr_s[i]) for i in range(len(FEATURE_COLS))}

        md, rel_risk, bucket = verbal_prediction(standardized, summary)
        st.markdown(md)

        input_df = pd.DataFrame([standardized])
        partial_hazard = float(cph.predict_partial_hazard(input_df)[0])
        st.markdown(f"**Cox predicted partial hazard:** `{partial_hazard:.3f}` (relative risk scale)")

st.divider()

# =========================
# SECTION 2: VISUALIZE (dropdown)
# =========================
st.header("2) Visualize")
st.caption("Choose a graph from the dropdown. Only one chart is shown at a time for readability.")

# Optional: preview in expander
with st.expander("Preview data + stats", expanded=False):
    st.write(f"Showing first {MAX_PREVIEW_ROWS} rows:")
    st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True)
    st.subheader("Summary Statistics")
    st.dataframe(df[FEATURE_COLS + [DURATION_COL, EVENT_COL]].describe(), use_container_width=True)

graph_options = [
    "Feature distribution (histogram)",
    "Feature correlation heatmap",
    "Kaplan–Meier survival (overall)",
    "Kaplan–Meier survival (high vs low, choose feature)",
    "Cox hazard ratios (bar, log scale)",
    "Predicted risk distribution (partial hazard)",
]
graph_choice = st.selectbox("Select a graph to display", graph_options, index=0)

# ---- Graph 1: histogram
if graph_choice == "Feature distribution (histogram)":
    c1, c2 = st.columns([1, 1])
    with c1:
        feat = st.selectbox("Feature", FEATURE_COLS, index=0)
    with c2:
        bins = st.slider("Bins", min_value=10, max_value=80, value=30, step=5)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(df[feat].dropna().values, bins=bins)
    ax.set_title(f"Distribution: {feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    st.pyplot(fig, use_container_width=True)

# ---- Graph 2: correlation heatmap (annotated)
elif graph_choice == "Feature correlation heatmap":
    corr = df[FEATURE_COLS].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
    ax.set_yticklabels(FEATURE_COLS)
    ax.set_title("Feature Correlations")

    for i in range(len(FEATURE_COLS)):
        for j in range(len(FEATURE_COLS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)

# ---- Graph 3: KM overall
elif graph_choice == "Kaplan–Meier survival (overall)":
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    kmf.fit(durations=df[DURATION_COL], event_observed=df[EVENT_COL], label="All cells")
    kmf.plot(ax=ax, ci_show=True)
    ax.set_title("Overall Survival Curve")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Survival probability")
    st.pyplot(fig, use_container_width=True)

# ---- Graph 4: KM split
elif graph_choice == "Kaplan–Meier survival (high vs low, choose feature)":
    split_feat = st.selectbox("Split feature", FEATURE_COLS, index=FEATURE_COLS.index("microglia_contact_time"))
    med = df[split_feat].median()
    high = df[df[split_feat] >= med]
    low = df[df[split_feat] < med]

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    kmf.fit(low[DURATION_COL], low[EVENT_COL], label=f"Low {split_feat} (< median)")
    kmf.plot(ax=ax, ci_show=True)
    kmf.fit(high[DURATION_COL], high[EVENT_COL], label=f"High {split_feat} (≥ median)")
    kmf.plot(ax=ax, ci_show=True)

    ax.set_title(f"Survival by {split_feat} (Median Split)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Survival probability")
    st.pyplot(fig, use_container_width=True)

# ---- Graph 5: Cox hazard ratios
elif graph_choice == "Cox hazard ratios (bar, log scale)":
    hr = summary["exp(coef)"].copy()

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(hr.index, hr.values)
    ax.axhline(1.0)
    ax.set_yscale("log")
    ax.set_title("Hazard Ratios (log scale) — >1 increases risk, <1 protective")
    ax.set_ylabel("Hazard ratio (exp(coef))")
    ax.set_xticklabels(hr.index, rotation=25, ha="right")
    st.pyplot(fig, use_container_width=True)

# ---- Graph 6: risk distribution
elif graph_choice == "Predicted risk distribution (partial hazard)":
    partial_haz = cph.predict_partial_hazard(df_scaled[FEATURE_COLS]).values.ravel()
    bins = st.slider("Bins", min_value=10, max_value=120, value=40, step=5)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.hist(partial_haz, bins=bins)
    ax.set_title("Distribution of Predicted Partial Hazard (Relative Risk)")
    ax.set_xlabel("Partial hazard")
    ax.set_ylabel("Count")
    st.pyplot(fig, use_container_width=True)

st.divider()

# =========================
# SECTION 3: CHAT
# =========================
st.header("3) Chat")
st.caption("Local assistant uses the Cox summary (no external API).")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

user_msg = st.chat_input("Ask about hazard ratios, p-values, or specific features...")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    reply = simple_local_chat(user_msg, summary)

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.chat.append(("assistant", reply))

st.caption("Note: Cox models predict relative hazard over time. Exact time prediction needs calibration or additional modeling.")