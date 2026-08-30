from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int32, Int64


# 1. Define the customer entity
customer = Entity(
    name="customer",
    join_keys=["nameOrig"],
    value_type=ValueType.STRING,
    description="Customer who starts a financial transaction",
)

# 2. Define the Parquet feature source
transaction_source = FileSource(
    name="transaction_feature_source",
    path="../../data/transaction_features.parquet",
    timestamp_field="event_timestamp",
)


# 3. Define the customer Feature View
customer_transaction_features = FeatureView(
    name="customer_transaction_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(
            name="txn_count_so_far",
            dtype=Int64,
        ),
        Field(
            name="total_sent_so_far",
            dtype=Float64,
        ),
        Field(
            name="avg_amount_so_far",
            dtype=Float64,
        ),
        Field(
            name="last_txn_amount",
            dtype=Float64,
        ),
        Field(
            name="hours_since_last_txn",
            dtype=Float64,
        ),
        Field(
            name="balance_mismatch",
            dtype=Int32,
        ),
    ],
    source=transaction_source,
    description="Historical customer transaction features for fraud detection",
)