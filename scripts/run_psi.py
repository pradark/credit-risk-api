import pandas as pd

from app.monitoring.psi import calculate_feature_psi


reference = pd.read_parquet(
    "monitoring/reference/reference_data.parquet"
)


production = pd.read_parquet(
    "production_test.parquet"
)


psi_results = calculate_feature_psi(
    reference,
    production
)


print("\nPSI Results")
print("-" * 50)

print(
    psi_results.to_string(
        index=False
    )
)