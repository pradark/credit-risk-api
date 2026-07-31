import pandas as pd
import numpy as np


np.random.seed(123)

n = 5000


df = pd.DataFrame({

    "income": np.random.normal(
        72000,
        26000,
        n
    ).clip(20000,200000),

    "fico": np.random.normal(
        690,
        55,
        n
    ).clip(300,850),

    "loan_amount": np.random.normal(
        26000,
        12000,
        n
    ).clip(1000,100000),

    "age": np.random.normal(
        42,
        11,
        n
    ).clip(18,80),

    "debt": np.random.normal(
        17000,
        9000,
        n
    ).clip(0,100000),

    "employment": np.random.normal(
        8,
        3,
        n
    ).clip(0,40),

    "balance": np.random.normal(
        14000,
        9000,
        n
    ).clip(0,100000),

    "utilization": np.random.beta(
        2.2,
        5,
        n
    ),

    "accounts": np.random.poisson(
        5,
        n
    ),

    "history": np.random.normal(
        9,
        3,
        n
    ).clip(0,30)

})


df.to_parquet(
    "production_test.parquet"
)

print(df.shape)