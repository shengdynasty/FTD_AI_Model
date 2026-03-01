# app.py
# Streamlit app: Upload CSV -> Fit Cox survival model -> Visualize data -> Verbal predictions -> Local chat assistant
#
# Run:
#   streamlit run app.py
#
# Expected CSV columns (survival mode):
#   neurite_growth, calcium_rate, aggregation_slope, microglia_contact_time,
#   time_to_failure_hours, event_observed
#
# Optional:
#   donor_id (not required for this app, but nice to have)

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter


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
MAX_HIST_BINS = 30


# =========================
# HELPERS
# =========================
def load_dataframe(uploaded_file, fallback_path: str) -> pd.DataFrame:
    """Load uploaded CSV if provided; otherwise load local fallback."""
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()

    if not os.path.exists(fallback_path):
        st.error(
            f"Could not find `{fallback_path}` and no CSV was uploaded.\n\n"
            "Upload a CSV using the sidebar, or place `cell_data.csv` in the same folder as app.py."
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
    """Readable summary of coefficients/hazard ratios."""
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
    """
    Turn a single sample into a natural language prediction.
    Returns: (markdown_text, relative_risk, bucket)
    """
    coefs = cox_summary["coef"].to_dict()

    # Contributions in Cox are coef * x (standardized)
    contribs = {k: coefs[k] * float(standardized_features[k]) for k in FEATURE_COLS}
    sorted_feats = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)

    lin_pred = sum(contribs.values())
    rel_risk = math.exp(lin_pred)

    # Friendly bucket thresholds (heuristic)
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
    """Local assistant (no API): answers using Cox summary + rule-based explanations."""
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

    if "hazard ratio" in msg or "hr" in msg or "what is hazard" in msg:
        return (
            "**Hazard ratio (HR)** is a relative risk over time.\n\n"
            "- HR > 1: higher values of that feature → faster failure\n"
            "- HR < 1: higher values of that feature → slower failure\n\n"
            "In a Cox model, these effects are *multiplicative* on the hazard."
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
st.caption("Survival modeling + visualizations + verbal interpretation + local chat (no API)")

# Sidebar: upload
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
df = load_dataframe(uploaded_file, DEFAULT_DATA_PATH)
st.sidebar.caption("Using: " + ("Uploaded CSV" if uploaded_file is not None else DEFAULT_DATA_PATH))

# Validate required columns
validate_columns(df)

# Clean
df = df.dropna(subset=FEATURE_COLS + [DURATION_COL, EVENT_COL]).copy()
df[EVENT_COL] = df[EVENT_COL].astype(int)

# Tabs
tab_predict, tab_viz, tab_chat = st.tabs(["Predict", "Visualizations", "Chat"])

# Fit scaler + Cox on full data (demo). For strict research, fit on train split only.
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = scaler.fit_transform(df_scaled[FEATURE_COLS])

cph = CoxPHFitter(penalizer=PENALIZER)
cph.fit(df_scaled[FEATURE_COLS + [DURATION_COL, EVENT_COL]],
        duration_col=DURATION_COL, event_col=EVENT_COL)

summary = cph.summary.loc[FEATURE_COLS, ["coef", "exp(coef)", "p"]]


# =========================
# TAB: PREDICT
# =========================
with tab_predict:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Cox Model Summary")
        st.dataframe(summary, use_container_width=True)
        st.markdown("#### Quick interpretation")
        st.markdown(describe_effects(summary))

    with right:
        st.subheader("Single-cell verbal prediction")
        st.write("Enter a cell’s **raw feature values** (the app standardizes them internally).")

        raw_inputs = {}
        raw_inputs["neurite_growth"] = st.number_input(
            "neurite_growth", value=float(df["neurite_growth"].mean())
        )
        raw_inputs["calcium_rate"] = st.number_input(
            "calcium_rate", value=float(df["calcium_rate"].mean())
        )
        raw_inputs["aggregation_slope"] = st.number_input(
            "aggregation_slope", value=float(df["aggregation_slope"].mean())
        )
        raw_inputs["microglia_contact_time"] = st.number_input(
            "microglia_contact_time", value=float(df["microglia_contact_time"].mean())
        )

        if st.button("Predict risk (verbal)"):
            arr = np.array([[raw_inputs[c] for c in FEATURE_COLS]])
            arr_s = scaler.transform(arr)[0]
            standardized = {FEATURE_COLS[i]: float(arr_s[i]) for i in range(len(FEATURE_COLS))}

            md, rel_risk, bucket = verbal_prediction(standardized, summary)
            st.markdown(md)

            # Cox partial hazard (relative hazard). cph expects standardized features.
            input_df = pd.DataFrame([standardized])
            partial_hazard = float(cph.predict_partial_hazard(input_df)[0])
            st.markdown(f"**Cox predicted partial hazard:** `{partial_hazard:.3f}` (relative risk scale)")


# =========================
# TAB: VISUALIZATIONS
# =========================
with tab_viz:
    st.subheader("Dataset Preview")
    st.write(f"Showing first {MAX_PREVIEW_ROWS} rows:")
    st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df[FEATURE_COLS + [DURATION_COL, EVENT_COL]].describe(), use_container_width=True)

    st.subheader("Feature Distributions")
    for col in FEATURE_COLS:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna().values, bins=MAX_HIST_BINS)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap (Features)")
    corr = df[FEATURE_COLS].corr(numeric_only=True)
    fig, ax = plt.subplots()
    im = ax.imshow(corr.values)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
    ax.set_yticklabels(FEATURE_COLS)
    fig.colorbar(im)
    ax.set_title("Feature Correlations")
    st.pyplot(fig)

    st.subheader("Kaplan–Meier Survival Curve (Overall)")
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()
    kmf.fit(durations=df[DURATION_COL], event_observed=df[EVENT_COL], label="All cells")
    kmf.plot(ax=ax)
    ax.set_title("Overall Survival Curve")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Survival probability")
    st.pyplot(fig)

    st.subheader("Kaplan–Meier: High vs Low Microglia Contact (Median Split)")
    med = df["microglia_contact_time"].median()
    high = df[df["microglia_contact_time"] >= med]
    low = df[df["microglia_contact_time"] < med]

    fig, ax = plt.subplots()
    kmf.fit(low[DURATION_COL], low[EVENT_COL], label="Low microglia contact")
    kmf.plot(ax=ax)
    kmf.fit(high[DURATION_COL], high[EVENT_COL], label="High microglia contact")
    kmf.plot(ax=ax)
    ax.set_title("Survival by Microglia Contact")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Survival probability")
    st.pyplot(fig)

    st.subheader("Cox Hazard Ratios (Feature Effects)")
    hr = summary["exp(coef)"].copy()
    fig, ax = plt.subplots()
    ax.bar(hr.index, hr.values)
    ax.axhline(1.0)
    ax.set_yscale("log")
    ax.set_title("Hazard Ratios (log scale) — >1 increases risk, <1 protective")
    ax.set_ylabel("Hazard ratio (exp(coef))")
    ax.set_xticklabels(hr.index, rotation=45, ha="right")
    st.pyplot(fig)

    st.subheader("Predicted Risk Distribution (Partial Hazard)")
    # Using standardized features because Cox was fit on df_scaled
    partial_haz = cph.predict_partial_hazard(df_scaled[FEATURE_COLS]).values.ravel()

    fig, ax = plt.subplots()
    ax.hist(partial_haz, bins=40)
    ax.set_title("Distribution of Predicted Partial Hazard (Relative Risk)")
    ax.set_xlabel("Partial hazard")
    ax.set_ylabel("Count")
    st.pyplot(fig)


# =========================
# TAB: CHAT
# =========================
with tab_chat:
    st.subheader("Chat with the model assistant")
    st.caption("This assistant answers using the Cox model summary (no external API).")

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

st.caption("Note: Cox models predict relative hazard over time. Exact time prediction requires calibration or additional modeling.")