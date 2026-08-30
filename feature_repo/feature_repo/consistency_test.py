from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore


REPO_DIR = Path(__file__).parent
PROJECT_DATA_DIR = REPO_DIR.parent.parent / "data"

feature_path = PROJECT_DATA_DIR / "transaction_features.parquet"


FEATURE_NAMES = [
    "txn_count_so_far",
    "total_sent_so_far",
    "avg_amount_so_far",
    "last_txn_amount",
    "hours_since_last_txn",
    "balance_mismatch",
]


FEATURE_REFERENCES = [
    f"customer_transaction_features:{feature_name}"
    for feature_name in FEATURE_NAMES
]


# Connect to Feast
store = FeatureStore(repo_path=str(REPO_DIR))


# Load the prepared offline feature data
feature_df = pd.read_parquet(feature_path)


# Find the latest feature row for every customer
latest_customer_rows = (
    feature_df
    .sort_values("event_timestamp")
    .drop_duplicates(subset=["nameOrig"], keep="last")
)


# Select customers with some transaction history
selected_rows = (
    latest_customer_rows[
        latest_customer_rows["txn_count_so_far"] > 0
    ]
    .head(5)
    .copy()
)


# Fallback if necessary
if len(selected_rows) == 0:
    selected_rows = latest_customer_rows.head(5).copy()


print("Customers selected for the consistency test:")
print(selected_rows[["nameOrig", "event_timestamp"]])

# Feast requires an entity DataFrame for historical retrieval
offline_entity_df = selected_rows[
    ["nameOrig", "event_timestamp"]
].copy()


offline_df = store.get_historical_features(
    entity_df=offline_entity_df,
    features=FEATURE_REFERENCES,
).to_df()


print("\nOffline feature values:")
print(offline_df)

online_entity_rows = [
    {"nameOrig": customer_id}
    for customer_id in selected_rows["nameOrig"].tolist()
]


online_response = store.get_online_features(
    features=FEATURE_REFERENCES,
    entity_rows=online_entity_rows,
)


online_df = pd.DataFrame(online_response.to_dict())


print("\nOnline feature values:")
print(online_df)

# Keep only the columns needed for comparison
offline_comparison = offline_df[
    ["nameOrig"] + FEATURE_NAMES
].copy()

online_comparison = online_df[
    ["nameOrig"] + FEATURE_NAMES
].copy()


# Match offline and online rows using the customer ID
comparison_df = offline_comparison.merge(
    online_comparison,
    on="nameOrig",
    suffixes=("_offline", "_online"),
)


comparison_results = {}


for feature in FEATURE_NAMES:
    offline_values = pd.to_numeric(
        comparison_df[f"{feature}_offline"],
        errors="coerce",
    )

    online_values = pd.to_numeric(
        comparison_df[f"{feature}_online"],
        errors="coerce",
    )

    matches = np.isclose(
        offline_values,
        online_values,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )

    comparison_df[f"{feature}_matches"] = matches
    comparison_results[feature] = bool(matches.all())


print("\nConsistency results:")

for feature, matches in comparison_results.items():
    status = "PASS" if matches else "FAIL"
    print(f"{feature}: {status}")


all_features_match = all(comparison_results.values())

print("\nAll features consistent:", all_features_match)


assert all_features_match, (
    "Offline and online feature values are not consistent."
)

print("\nStep 7 completed successfully!")