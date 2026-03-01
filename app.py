import os
import math
import pandas as pd
import numpy as np
import streamlit as st

from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

# Optional: OpenAI chat 
"""
USE_OPENAI = False
try:
    from openai import OpenAI
    USE_OPENAI = True
except Exception:
    USE_OPENAI = False

"""
# =========================
# CONFIG
# =========================
DATA_PATH = "cell_data.csv"

FEATURE_COLS = [
    "neurite_growth",
    "calcium_rate",
    "aggregation_slope",
    "microglia_contact_time",
]

DURATION_COL = "time_to_failure_hours"
EVENT_COL = "event_observed"
GROUP_COL = "donor_id"

# Cox model penalizer for stability
PENALIZER = 0.05


# =========================
# HELPER FUNCTIONS
# =========================
def safe_sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def describe_effects(cox_summary: pd.DataFrame) -> str:
    """
    Create a readable summary of coefficients/hazard ratios.
    """
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


def verbal_prediction(input_features: dict, cox_summary: pd.DataFrame) -> str:
    """
    Turn a single sample into a natural language prediction.
    We use:
      - risk score approximation: sum(coef * standardized_feature)
      - top contributing factors (by absolute contribution)
    Note: Cox model predicts relative risk (hazard), not exact time by itself.
    """

    # Get coefficients
    coefs = cox_summary["coef"].to_dict()

    # Compute contributions (we assume input features are already standardized)
    contribs = {k: coefs[k] * input_features[k] for k in FEATURE_COLS}
    sorted_feats = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)

    # Relative risk proxy = exp(sum(contrib))
    lin_pred = sum(contribs.values())
    rel_risk = math.exp(lin_pred)

    # Convert to a friendly bucket (purely descriptive)
    # (These thresholds are heuristic—good enough for a demo.)
    if rel_risk >= 2.0:
        bucket = "HIGH"
    elif rel_risk >= 1.2:
        bucket = "MODERATE"
    else:
        bucket = "LOW"

    top_factors = []
    for feat, val in sorted_feats[:3]:
        direction = "raises" if val > 0 else "lowers"
        top_factors.append(f"{feat} **{direction}** risk")

    text = f"""
### Verbal risk prediction
**Relative risk (hazard) estimate:** `{rel_risk:.2f}×` baseline  
**Risk bucket:** **{bucket}** (higher = likely faster failure)

**Main drivers (top 3):**
- {top_factors[0]}
- {top_factors[1]}
- {top_factors[2]}

**Interpretation tip:**  
- Positive contributions push toward **faster failure**  
- Negative contributions push toward **slower failure**
"""
    return text


def simple_local_chat(user_msg: str, cox_summary: pd.DataFrame) -> str:
    """
    A no-API 'assistant': answers using the Cox summary + a few rule-based responses.
    """
    msg = user_msg.lower()

    if "most important" in msg or "strongest" in msg or "top" in msg:
        # strongest by absolute coef
        s = cox_summary.copy()
        s["abs_coef"] = s["coef"].abs()
        top = s.sort_values("abs_coef", ascending=False).head(3)
        items = [f"{i} (HR={top.loc[i,'exp(coef)']:.2f})" for i in top.index]
        return f"The strongest predictors (largest absolute effect) are: {', '.join(items)}."

    if "microglia" in msg:
        hr = float(cox_summary.loc["microglia_contact_time", "exp(coef)"])
        return (
            f"Microglia contact time has HR={hr:.2f}. That means higher microglia contact is associated "
            f"with faster failure (higher hazard)."
        )

    if "aggregation" in msg or "tdp-43" in msg or "tau" in msg:
        hr = float(cox_summary.loc["aggregation_slope", "exp(coef)"])
        return (
            f"Aggregation slope has HR={hr:.2f}. Higher early aggregation trend predicts faster failure."
        )

    if "calcium" in msg:
        hr = float(cox_summary.loc["calcium_rate", "exp(coef)"])
        return (
            f"Calcium rate has HR={hr:.2f}. Because HR < 1, higher calcium activity is protective (slower failure)."
        )

    if "neurite" in msg:
        hr = float(cox_summary.loc["neurite_growth", "exp(coef)"])
        return (
            f"Neurite growth has HR={hr:.2f}. HR slightly below 1 suggests healthier neurite growth is mildly protective."
        )

    if "what is hazard" in msg or "hazard ratio" in msg:
        return (
            "Hazard ratio (HR) describes relative risk of failure over time. HR>1 means higher feature values are linked "
            "to faster failure; HR<1 means they are linked to slower failure."
        )

    return (
        "Ask me things like: 'Which feature is most important?', "
        "'Explain microglia_contact_time', or 'What does HR mean?'"
    )


def openai_chat(user_msg: str, cox_summary: pd.DataFrame) -> str:
    """
    Optional: real AI assistant. Requires OPENAI_API_KEY env var and openai package.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI chat is not enabled because OPENAI_API_KEY is not set."

    client = OpenAI(api_key=api_key)

    context = "Cox model summary (coef, HR, p):\n" + cox_summary.to_string()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant interpreting an FTD cell-level Cox survival model."},
            {"role": "user", "content": f"{context}\n\nUser question: {user_msg}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="FTD Cell Failure Predictor", layout="wide")
st.title("FTD Cell-Level Failure Predictor (Survival + Verbal Interpretation + Chat)")

# Load data
# =========================
# DATA LOADING (UPLOAD OR LOCAL)
# =========================
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

def load_dataframe(uploaded_file, fallback_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
    else:
        if not os.path.exists(fallback_path):
            st.error(
                f"Could not find `{fallback_path}` and no CSV was uploaded.\n\n"
                f"Upload a CSV using the sidebar, or place `{fallback_path}` in the same folder as app.py."
            )
            st.stop()
        try:
            return pd.read_csv(fallback_path)
        except Exception as e:
            st.error(f"Failed to read `{fallback_path}`: {e}")
            st.stop()

df = load_dataframe(uploaded_file, DATA_PATH)

st.sidebar.caption("Using: " + ("Uploaded CSV" if uploaded_file is not None else DATA_PATH))

df = pd.read_csv(DATA_PATH)

required = FEATURE_COLS + [DURATION_COL, EVENT_COL]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(
        "Uploaded CSV is missing required columns:\n\n"
        + "\n".join([f"- {c}" for c in missing])
        + "\n\nExpected columns include:\n"
        + "\n".join([f"- {c}" for c in required])
    )
    st.stop()

df = df.dropna(subset=required).copy()
df[EVENT_COL] = df[EVENT_COL].astype(int)

# Fit scaler on full dataset (demo app). For strict science, fit on train split only.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURE_COLS])
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = X_scaled

# Fit Cox model
cph = CoxPHFitter(penalizer=PENALIZER)
cph.fit(df_scaled[FEATURE_COLS + [DURATION_COL, EVENT_COL]], duration_col=DURATION_COL, event_col=EVENT_COL)
summary = cph.summary.loc[FEATURE_COLS, ["coef", "exp(coef)", "p"]]

# Layout
left, right = st.columns([1, 1])

with left:
    st.subheader("Model summary (Cox)")
    st.dataframe(summary)

    st.markdown("#### Quick interpretation")
    st.markdown(describe_effects(summary))

with right:
    st.subheader("Make a prediction for a single cell")

    # User inputs in raw units (we will standardize using the dataset scaler)
    raw_inputs = {}
    raw_inputs["neurite_growth"] = st.number_input("neurite_growth", value=float(df["neurite_growth"].mean()))
    raw_inputs["calcium_rate"] = st.number_input("calcium_rate", value=float(df["calcium_rate"].mean()))
    raw_inputs["aggregation_slope"] = st.number_input("aggregation_slope", value=float(df["aggregation_slope"].mean()))
    raw_inputs["microglia_contact_time"] = st.number_input("microglia_contact_time", value=float(df["microglia_contact_time"].mean()))

    if st.button("Predict risk (verbal)"):
        # Standardize inputs using fitted scaler
        arr = np.array([[raw_inputs[c] for c in FEATURE_COLS]])
        arr_s = scaler.transform(arr)[0]
        standardized = {FEATURE_COLS[i]: float(arr_s[i]) for i in range(len(FEATURE_COLS))}

        st.markdown(verbal_prediction(standardized, summary))

        # Show partial hazard (relative hazard) directly from Cox model too
        input_df = pd.DataFrame([standardized])
        # Note: cph expects same columns; these are standardized features
        partial_hazard = float(cph.predict_partial_hazard(input_df)[0])
        st.markdown(f"**Cox predicted partial hazard:** `{partial_hazard:.3f}` (relative risk scale)")

st.divider()

# Chat section
st.subheader("Chat with the model assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

user_msg = st.chat_input("Ask about the model, features, hazard ratios, or results...")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        if USE_OPENAI and os.getenv("OPENAI_API_KEY"):
            reply = openai_chat(user_msg, summary)
        else:
            reply = simple_local_chat(user_msg, summary)

        st.markdown(reply)

    st.session_state.chat.append(("assistant", reply))

st.caption("Note: Cox models predict relative hazard (risk over time). Exact time prediction requires extra assumptions or calibration.")