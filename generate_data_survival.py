import numpy as np
import pandas as pd

np.random.seed(42)

NUM_DONORS = 8
CELLS_PER_DONOR = 300

# Observation window: 7 days in hours
MAX_HOURS = 7 * 24  # 168

rows = []

for donor in range(NUM_DONORS):
    donor_risk_bias = np.random.normal(0, 0.6)

    for cell in range(CELLS_PER_DONOR):
        neurite_growth = np.random.normal(0.2, 0.05) - 0.02 * donor_risk_bias
        calcium_rate = np.random.normal(5, 1.0) - 0.5 * donor_risk_bias
        aggregation_slope = np.random.normal(0.01, 0.005) + 0.01 * donor_risk_bias
        microglia_contact_time = np.random.normal(15, 5) + 4 * donor_risk_bias

        # Risk score (higher -> faster failure)
        risk_score = (
            2.8 * aggregation_slope +
            0.09 * microglia_contact_time -
            0.45 * neurite_growth -
            0.18 * calcium_rate +
            donor_risk_bias
        )

        # Convert risk_score -> hazard rate (positive)
        # This produces realistic variation in failure times.
        hazard_rate = np.exp(risk_score)  # >0

        # Sample time-to-event from exponential distribution
        # Higher hazard -> shorter expected time
        # Add scaling so times land in a reasonable range
        scale = 80.0
        time_to_failure = np.random.exponential(scale=scale / (hazard_rate + 1e-9))

        # Censor at 7 days (168h)
        event_observed = 1 if time_to_failure <= MAX_HOURS else 0
        observed_time = min(time_to_failure, MAX_HOURS)

        # For compatibility with your existing binary model:
        failed_7d = event_observed

        rows.append([
            f"D{donor}",
            neurite_growth,
            calcium_rate,
            aggregation_slope,
            microglia_contact_time,
            failed_7d,
            observed_time,
            event_observed
        ])

df = pd.DataFrame(rows, columns=[
    "donor_id",
    "neurite_growth",
    "calcium_rate",
    "aggregation_slope",
    "microglia_contact_time",
    "failed_7d",
    "time_to_failure_hours",
    "event_observed"
])

df.to_csv("cell_data.csv", index=False)

print("Created: cell_data.csv")
print("Rows:", len(df))
print("Event rate:", df["event_observed"].mean())
print("Median observed time (hours):", df["time_to_failure_hours"].median())