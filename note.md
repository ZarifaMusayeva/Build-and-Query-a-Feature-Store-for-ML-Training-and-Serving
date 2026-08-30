# Project Notes: Feature Store for Fraud Detection

This document covers the design decisions, trade-offs, and architecture for the entire project — both the feature engineering pipeline (Data & Features) and the Feast feature store setup (Feast & Validation).

---

## 1. Dataset and problem framing

We use PaySim, a synthetic mobile-money transaction log of ~6.35M transactions over a simulated 30-day period. The task is to build a feature store that serves the same fraud-detection features consistently at both training time (offline, point-in-time-correct) and inference time (online, low-latency), with no leakage of future information into any feature.

- **Entity:** `nameOrig` — the sending customer, since fraud behavior is a property of the sender's account.
- **Label:** `isFraud`.
- **Event time:** `step`, converted into an absolute timestamp (see §3).

---

## 2. Sampling strategy

The full dataset (6.35M rows, ~470MB) is too large to work with comfortably for pipeline development, and GitHub cannot host a file that size in any case. Rather than a random transaction-level sample — which would fragment customer histories and make historical features meaningless — we used **customer-level sampling**:

- All customers associated with at least one fraudulent transaction are kept in full.
- 3,000 additional customers are randomly sampled from the non-fraud population, also kept in full (their complete transaction history, not just a snapshot).

This preserves complete per-customer histories, which historical features depend on, and guarantees the subset still contains fraud examples for later point-in-time validation.

**Trade-off / limitation:** this is a **label-aware sampling strategy** — customer selection is informed by `isFraud`. This does not leak information into the *features themselves* (features are still computed purely from prior transaction history), but it does mean the subset's fraud rate (much higher than ~0.129%) is not representative of the true population, and the subset should not be used to estimate real-world fraud prevalence. It is intended purely for feature-pipeline development and Feast validation.

**Data characteristic discovered during validation:** in PaySim, `nameOrig` values repeat extremely rarely — about 99.85% of senders appear in only one transaction in the entire dataset, and the maximum repeat count is 3. This means most "so-far" historical features are legitimately zero for most transactions; this is a genuine property of the data, not a pipeline defect, and it directly affects which customers are useful for demonstrating multi-transaction point-in-time behavior (both in the feature notebook and in Feast's online/offline consistency tests, customers with `txn_count_so_far > 0` are deliberately prioritized).

---

## 3. Event time conversion

PaySim's `step` field represents simulated hours (1 step = 1 hour), not a real-world timestamp. Feast requires a genuine timestamp column for point-in-time joins, so `step` is converted via:

```python
event_timestamp = base_date + step (as hours)
```

The reference date (`2024-01-01`) is arbitrary and does not represent an actual transaction date — only the relative ordering between events matters for point-in-time correctness.

---

## 4. Feature design and leakage prevention

All historical features are computed so that a transaction only ever "sees" information from **strictly prior** points in time.

### The same-step leakage problem

A row-based `shift(1)` (shifting by transaction row, sorted by timestamp) is not safe here: PaySim allows multiple transactions from the same sender within the same `step` (i.e., the same simulated hour). If two transactions share a `step`, they share the same `event_timestamp`, and there is no way to know their true order within that hour. A row-based shift would treat one as "before" the other, which is not a claim the data actually supports — this is a subtle point-in-time leakage risk.

### Solution: step-level aggregation

Features are aggregated at the `(nameOrig, step)` level first:
1. Compute per-step transaction count and amount sum for each customer.
2. Take a cumulative sum **including** the current step, then subtract the current step's own contribution. What remains reflects only strictly prior steps.
3. Merge these step-level features back onto every transaction row that shares that step — so all transactions within the same step receive identical (safe) historical values, since none of them can legitimately claim to precede the other.

### Feature definitions

| Feature | Meaning | Notes |
|---|---|---|
| `txn_count_so_far` | Number of transactions the sender made in all strictly prior steps | Historical, step-level |
| `avg_amount_so_far` | Average transaction amount across the sender's prior steps | Historical, step-level |
| `total_sent_so_far` | Cumulative transaction amount across the sender's prior steps | Historical, step-level |
| `last_txn_amount` | Amount from the sender's most recent *prior distinct* step | Historical; if that step had multiple transactions, the last one by row order is used as an approximation, since true intra-step order is unknown |
| `hours_since_last_txn` | Hours elapsed since the sender's previous distinct step | Historical, step-level |
| `balance_mismatch` | Flag for whether `oldbalanceOrg - amount` differs from `newbalanceOrig` | **Current-event feature**, not historical — computed from the transaction's own data, not the past. Also not "cleaning": inconsistent rows are kept, not dropped, because the inconsistency itself is a useful signal |

### Leakage validation

Three checks confirm point-in-time correctness before the data reaches Feast:
1. **First-step check** — every customer's first step has all historical features equal to zero.
2. **Same-step consistency check** — transactions sharing a `(nameOrig, step)` pair have identical historical feature values, proving no leakage occurs between transactions that share a timestamp.
3. **Manual cross-validation** — for a sample customer, features computed by hand from raw data (using only `step < current_step`) are compared against the pipeline's output and must match exactly.

---

## 5. Feast feature store setup

### Entity, source, and feature view

- **Entity:** `customer`, joined on `nameOrig`.
- **Offline source:** a `FileSource` pointing at `transaction_features.parquet`, with `event_timestamp` as the timestamp field.
- **FeatureView:** `customer_transaction_features`, listing all six features above with a 365-day TTL, registered via `feast apply`.

### Offline store: Parquet

Chosen because it's simple, requires no external infrastructure, and is sufficient for point-in-time join semantics at this dataset scale. It would not scale to a production data warehouse setting (see §6).

### Online store: SQLite

Chosen for the same reason — a lightweight, file-based, dependency-free store that's enough to demonstrate low-latency lookup semantics locally, without standing up Redis or a managed key-value store.

### Point-in-time (offline) retrieval

`get_historical_features()` is called with the `entity_df` (customer + event_timestamp + label) prepared by the feature engineering pipeline. Feast performs the point-in-time join itself, matching each `(nameOrig, event_timestamp)` pair with the most recent feature values that existed strictly before that timestamp. The result is saved as `training_dataset.parquet` and validated to confirm no rows or expected columns are missing.

### Materialization and online retrieval

Features are materialized from the offline Parquet source into the SQLite online store. `get_online_features()` is then used to fetch each customer's *latest known* feature values — this is the same lookup pattern a real-time fraud-scoring service would use at inference time. Because `nameOrig` rarely repeats (see §2), customers with `txn_count_so_far > 0` are deliberately selected for these examples so that non-trivial (non-zero) feature values are actually exercised, with a fallback to any customer if fewer than 5 such customers exist in a given sample.

### Offline/online consistency check

For a shared set of customers, the same feature values are retrieved through both the offline path (`get_historical_features`, queried at each customer's latest known event timestamp) and the online path (`get_online_features`). The two results are merged and compared with `np.isclose` (rather than exact equality, to tolerate floating-point representation differences), and the test asserts that every feature matches for every customer. This is the project's core deliverable: proof that the feature values a model would train on offline are identical to what it would see at serving time, i.e., no training/serving skew.

**Trade-off to note:** the online store only ever holds the *latest materialized* value per entity — it has no native concept of "as of a specific past timestamp." A fair offline/online comparison therefore requires querying the offline path at (or very close to) the same timestamp used for the last materialization, rather than an arbitrary point in time.

---

## 6. Architectural trade-offs and what would change in production

- **File-based offline store → data warehouse.** A Parquet file offline store does not scale past a single machine. In production this would be replaced by BigQuery, Snowflake, or Redshift, as Feast supports natively.
- **SQLite online store → managed key-value store.** SQLite is fine for local development but would be replaced by Redis, DynamoDB, or a similar managed store for real low-latency, concurrent production traffic.
- **Batch materialization → scheduled/streaming ingestion.** Materialization here is a manual, one-off step. In production this would run on a schedule (e.g., via Airflow) or be fed by a streaming source for fresher online features.
- **Local file registry → remote/SQL registry.** The Feast registry here is a local file; production setups typically use a remote registry (S3/GCS-backed or SQL-backed) with CI/CD to propagate feature definition changes safely across dev/staging/prod.
- **Customer-level, label-aware sampling → full population.** For a real fraud model, feature computation would run over the entire population, not a fraud-oversampled subset; the subset here exists solely to make pipeline and point-in-time validation tractable and fraud examples visible during development.

---

## 7. Deliverables

- `notebooks/feature_engineering.ipynb` — exploration, cleaning, feature engineering, and leakage validation (Data & Features).
- `data/transaction_features.parquet`, `data/entity_df.parquet` — Feast offline source and entity dataframe.
- `feature_repo/feature_repo/` — Feast repository: entity/source/feature-view definitions, offline retrieval, online retrieval, and consistency test scripts (Feast & Validation).
- `data/training_dataset.parquet` — the point-in-time-correct training dataset produced by `historical_retrieval.py`.
- This document (`note.md`) and the project `README.md`.