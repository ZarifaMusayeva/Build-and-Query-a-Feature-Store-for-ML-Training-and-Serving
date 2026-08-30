# Feature Store for Fraud Detection (PaySim)

A reusable feature store, built with **Feast**, that serves the same transaction-level features consistently for both model training (offline) and real-time inference (online). Built on the [PaySim synthetic mobile-money dataset](https://www.kaggle.com/datasets/ealaxi/paysim1).

The goal is to avoid training/serving skew: features used at training time must match what the model sees at inference time, and no feature may leak information from after the prediction event.

## Project structure

```
feature-store-project/
├── data/                          # Raw CSV, feature Parquet, entity dataframe
│   ├── PS_20174392719_1491204439457_log.csv   (not tracked in git — see below)
│   ├── transaction_features.parquet
│   ├── entity_df.parquet
│   └── training_dataset.parquet   # output of the offline point-in-time retrieval
├── feature_repo/
│   └── feature_repo/
│       ├── feature_store.yaml           # Feast config: file offline store + SQLite online store
│       ├── feature_definitions.py       # Entity, FileSource, FeatureView definitions
│       ├── historical_retrieval.py      # Point-in-time training dataset retrieval
│       ├── online_retrieval.py          # Online (SQLite) feature lookup
│       ├── consistency_test.py          # Offline vs online consistency check
│       └── data/                        # Feast registry + SQLite online store (generated)
├── notebooks/
│   └── feature_engineering.ipynb  # Data exploration, cleaning, and feature engineering
└── note.md                        
```

## Dataset

- **Source:** PaySim, a synthetic mobile-money transaction log (~6.35M transactions, ~30 simulated days).
- **Entity key:** `nameOrig` (the sending customer).
- **Event time:** `step` (1 step = 1 simulated hour), converted to an absolute `event_timestamp` for Feast compatibility.
- The raw CSV (~470MB) is **not committed to this repository** (see `.gitignore`) since it exceeds GitHub's practical file-size limits. Download it from Kaggle and place it in `data/` before running the notebook.

## How the pipeline works

1. **Data & Features (`notebooks/feature_engineering.ipynb`)**
   Explores and cleans the raw dataset, selects a customer-level subset (all fraud-associated customers + 3,000 randomly sampled clean customers), converts `step` into `event_timestamp`, and computes leakage-safe historical features aggregated at the `(nameOrig, step)` level. Outputs `transaction_features.parquet` and `entity_df.parquet` into `data/`. See `note.md` for full design rationale.

2. **Feast setup (`feature_repo/`)**
   Defines the `customer` entity, a `FileSource` pointing at `transaction_features.parquet`, and a `FeatureView` (`customer_transaction_features`) listing all six features. Registered via `feast apply`.

3. **Offline retrieval (`historical_retrieval.py`)**
   Loads `entity_df.parquet` and calls `get_historical_features()` to produce a point-in-time-correct training dataset (`training_dataset.parquet`), joining each entity/timestamp pair with the feature values that were valid at that moment — never using information from the future.

4. **Materialization & online retrieval (`online_retrieval.py`)**
   Features are materialized into a local SQLite online store. `get_online_features()` then retrieves the latest known feature values for a given customer with low latency, simulating a real-time inference lookup.

5. **Consistency check (`consistency_test.py`)**
   Compares the offline path and the online path for the same set of customers and asserts that both return matching feature values — demonstrating there is no training/serving skew.

## Running it

```bash
# 1. Place the PaySim CSV in data/
# 2. Run the feature engineering notebook
jupyter notebook notebooks/feature_engineering.ipynb

# 3. Apply the Feast feature definitions
cd feature_repo/feature_repo
feast apply

# 4. Run point-in-time training retrieval
python historical_retrieval.py

# 5. Materialize features into the online store
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)

# 6. Run the online retrieval example
python online_retrieval.py

# 7. Run the offline/online consistency check
python consistency_test.py
```

## Features

| Feature | Meaning |
|---|---|
| `txn_count_so_far` | Number of prior transactions by the sender, computed at the step level |
| `avg_amount_so_far` | Average amount of the sender's prior transactions |
| `total_sent_so_far` | Cumulative amount of the sender's prior transactions |
| `last_txn_amount` | Amount from the sender's most recent prior step |
| `hours_since_last_txn` | Hours elapsed since the sender's previous distinct step |
| `balance_mismatch` | Current-event flag for inconsistent post-transaction balance |

Full definitions, calculation logic, and leakage-prevention rationale are documented in `note.md` and in the feature engineering notebook.