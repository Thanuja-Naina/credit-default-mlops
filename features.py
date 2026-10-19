"""Shared feature code. Used by BOTH train.py and app.py so training and serving stay identical."""
import numpy as np

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]        # repayment status per month
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]                         # bill amount per month
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]                       # amount paid per month
CATEGORY_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
RAW_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS


def add_features(df):
    """Create new columns that describe credit behaviour better than raw numbers."""
    df = df.copy()
    df["utilization"] = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(-1, 5)        # how much of the limit is used
    df["avg_bill"] = df[BILL_COLS].mean(axis=1)
    df["total_paid"] = df[PAY_AMT_COLS].sum(axis=1)
    total_bill = df[BILL_COLS].clip(lower=0).sum(axis=1).replace(0, np.nan)
    df["repay_ratio"] = (df["total_paid"] / total_bill).clip(0, 5).fillna(0)   # paid / billed
    df["max_delay"] = df[PAY_COLS].max(axis=1)                                 # worst payment delay
    df["months_delayed"] = (df[PAY_COLS] > 0).sum(axis=1)                      # how many months were late
    # some category codes are undocumented in the original data -> group them as "other"
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2]), 3)
    return df


NEW_COLS = ["utilization", "avg_bill", "total_paid", "repay_ratio", "max_delay", "months_delayed"]
NUMBER_COLS = [c for c in RAW_COLS if c not in CATEGORY_COLS] + NEW_COLS
