import pandas as pd
import numpy as np


np.random.seed(42)

n = 10000

df = pd.DataFrame({

    "income": np.random.normal(
        70000,
        25000,
        n
    ).clip(20000,200000),

    "fico": np.random.normal(
        700,
        50,
        n
    ).clip(300,850),

    "loan_amount": np.random.normal(
        25000,
        10000,
        n
    ).clip(1000,100000),

    "age": np.random.normal(
        40,
        10,
        n
    ).clip(18,80),

    "debt": np.random.normal(
        15000,
        8000,
        n
    ).clip(0,100000),

    "employment": np.random.normal(
        7,
        3,
        n
    ).clip(0,40),

    "balance": np.random.normal(
        12000,
        8000,
        n
    ).clip(0,100000),

    "utilization": np.random.beta(
        2,
        5,
        n
    ),

    "accounts": np.random.poisson(
        5,
        n
    ),

    "history": np.random.normal(
        8,
        3,
        n
    ).clip(0,30)

})


df.to_parquet(
    "monitoring/reference/reference_data.parquet"
)

print(df.head())
print(df.shape)
