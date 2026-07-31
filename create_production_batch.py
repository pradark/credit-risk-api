import pandas as pd
import numpy as np

reference = pd.read_parquet(
    "monitoring/reference/reference_data.parquet"
)

np.random.seed(42)

# Sample production applications
production = reference.sample(
    n=500,
    replace=True
).reset_index(drop=True)


# Simulate small production drift
# New applicants have slightly higher utilization
production["utilization"] = (
    production["utilization"] * 1.15
).clip(0, 1)


# Slight employment profile shift
production["employment"] = (
    production["employment"] * 0.90
).round()


# Save locally
production.to_parquet(
    "production_batch.parquet",
    index=False
)

print(production.head())
print()
print("Shape:", production.shape)
