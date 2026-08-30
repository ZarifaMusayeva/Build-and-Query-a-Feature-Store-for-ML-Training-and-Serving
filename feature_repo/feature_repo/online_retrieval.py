from pathlib import Path

import pandas as pd
from feast import FeatureStore


# Locate the Feast repository and prepared feature data
REPO_DIR = Path(__file__).parent
PROJECT_DATA_DIR = REPO_DIR.parent.parent / "data"

feature_path = PROJECT_DATA_DIR / "transaction_features.parquet"


# Connect to Feast
store = FeatureStore(repo_path=str(REPO_DIR))


# Load the Parquet file only to select example customer IDs
feature_df = pd.read_parquet(feature_path)

latest_customer_rows = (
    feature_df
    .sort_values("event_timestamp")
    .drop_duplicates(subset=["nameOrig"], keep="last")
)


# Prefer customers who have previous transaction history
selected_customers = (
    latest_customer_rows
    .loc[
        latest_customer_rows["txn_count_so_far"] > 0,
        "nameOrig",
    ]
    .head(5)
    .tolist()
)


# Fallback in case there are fewer than five repeated customers
if len(selected_customers) < 5:
    extra_customers = (
        latest_customer_rows[
            ~latest_customer_rows["nameOrig"].isin(selected_customers)
        ]["nameOrig"]
        .head(5 - len(selected_customers))
        .tolist()
    )

    selected_customers.extend(extra_customers)


print("Selected customers:")
print(selected_customers)


# Feast expects each entity as a dictionary
entity_rows = [
    {"nameOrig": customer_id}
    for customer_id in selected_customers
]


# Retrieve the latest features from SQLite
online_response = store.get_online_features(
    features=[
        "customer_transaction_features:txn_count_so_far",
        "customer_transaction_features:total_sent_so_far",
        "customer_transaction_features:avg_amount_so_far",
        "customer_transaction_features:last_txn_amount",
        "customer_transaction_features:hours_since_last_txn",
        "customer_transaction_features:balance_mismatch",
    ],
    entity_rows=entity_rows,
)


# Convert Feast's response into a DataFrame
online_df = pd.DataFrame(online_response.to_dict())

print("\nOnline features:")
print(online_df)