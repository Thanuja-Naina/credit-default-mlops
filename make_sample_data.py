"""Creates FAKE data with the same columns as the real dataset, only to test that the code runs.
NEVER report results from this fake data.   Run: python make_sample_data.py"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 3000
limit = rng.choice([20000, 50000, 120000, 300000], n)
df = pd.DataFrame({"LIMIT_BAL": limit, "SEX": rng.integers(1, 3, n), "EDUCATION": rng.integers(1, 5, n),
                   "MARRIAGE": rng.integers(1, 4, n), "AGE": rng.integers(21, 70, n)})
for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
    df[col] = rng.choice([-1, 0, 1, 2, 3], n, p=[0.2, 0.5, 0.15, 0.1, 0.05])
for i in range(1, 7):
    df[f"BILL_AMT{i}"] = (limit * rng.uniform(0, 1.1, n)).round()
    df[f"PAY_AMT{i}"] = (df[f"BILL_AMT{i}"] * rng.uniform(0, 0.6, n)).round()
risk = -1.5 + 0.8 * df["PAY_0"] + 0.9 * df["BILL_AMT1"] / df["LIMIT_BAL"] + rng.normal(0, 0.8, n)
df["default.payment.next.month"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-risk))).astype(int)
df.to_csv("data/sample_fake_data.csv", index=False)
print("Saved data/sample_fake_data.csv")
