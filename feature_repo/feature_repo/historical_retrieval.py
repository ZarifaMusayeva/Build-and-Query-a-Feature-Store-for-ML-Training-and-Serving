from pathlib import Path

import pandas as pd
from feast import FeatureStore


# The current folder contains feature_store.yaml
REPO_DIR = Path(__file__).parent

# Go back to feature_store_project/data
PROJECT_DATA_DIR = REPO_DIR.parent.parent / "data"

entity_path = PROJECT_DATA_DIR / "entity_df.parquet"
output_path = PROJECT_DATA_DIR / "training_dataset.parquet"


# Check whether the entity file exists
print("Entity file:", entity_path)
print("Entity file exists:", entity_path.exists())


# Load the prediction events
entity_df = pd.read_parquet(entity_path)

print("\nEntity data shape:", entity_df.shape)
print(entity_df.head())


# Connect to our Feast repository
store = FeatureStore(repo_path=str(REPO_DIR))


# Retrieve point-in-time-correct historical features
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_transaction_features:txn_count_so_far",
        "customer_transaction_features:total_sent_so_far",
        "customer_transaction_features:avg_amount_so_far",
        "customer_transaction_features:last_txn_amount",
        "customer_transaction_features:hours_since_last_txn",
        "customer_transaction_features:balance_mismatch",
    ],
).to_df()


print("\nTraining dataset shape:", training_df.shape)
print(training_df.head())


# Save the completed training dataset
training_df.to_parquet(output_path, index=False)

print("\nTraining dataset saved to:")
print(output_path)
expected_columns = [
    "nameOrig",
    "event_timestamp",
    "isFraud",
    "txn_count_so_far",
    "total_sent_so_far",
    "avg_amount_so_far",
    "last_txn_amount",
    "hours_since_last_txn",
    "balance_mismatch",
]

assert len(training_df) == len(entity_df)
assert all(column in training_df.columns for column in expected_columns)

print("\nMissing values:")
print(training_df.isna().sum())

print("\nStep 4 completed successfully!")