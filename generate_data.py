import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
NUM_DONORS = 8
CELLS_PER_DONOR = 300

rows = []

for donor in range(NUM_DONORS):

    # Donor-level biological variability
    donor_risk_bias = np.random.normal(0, 0.5)

    for cell in range(CELLS_PER_DONOR):

        # Early cellular features (0–24h window)
        neurite_growth = np.random.normal(0.2, 0.05) - 0.02 * donor_risk_bias
        calcium_rate = np.random.normal(5, 1.0) - 0.5 * donor_risk_bias
        aggregation_slope = np.random.normal(0.01, 0.005) + 0.01 * donor_risk_bias
        microglia_contact_time = np.random.normal(15, 5) + 4 * donor_risk_bias

        # Nonlinear biological risk model
        risk_score = (
            2.5 * aggregation_slope +
            0.08 * microglia_contact_time -
            0.4 * neurite_growth -
            0.15 * calcium_rate +
            donor_risk_bias
        )

        # Convert to probability using sigmoid
        failure_prob = 1 / (1 + np.exp(-risk_score))

        failed_7d = np.random.binomial(1, failure_prob)

        rows.append([
            f"D{donor}",
            neurite_growth,
            calcium_rate,
            aggregation_slope,
            microglia_contact_time,
            failed_7d
        ])

df = pd.DataFrame(rows, columns=[
    "donor_id",
    "neurite_growth",
    "calcium_rate",
    "aggregation_slope",
    "microglia_contact_time",
    "failed_7d"
])

df.to_csv("cell_data.csv", index=False)

print("Synthetic dataset created: cell_data.csv")
print("Rows:", len(df))
print("Failure rate:", df["failed_7d"].mean())