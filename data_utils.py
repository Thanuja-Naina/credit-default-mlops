"""Helpers for loading the credit-default dataset."""
import pandas as pd


def load_credit_data(path):
    """Load the standard CSV or a CSV exported with the header on its second row."""
    df = pd.read_csv(path)
    if "LIMIT_BAL" not in df.columns:
        df = pd.read_csv(path, header=1)

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "default.payment.next.month": "default",
        "default payment next month": "default",
    })
    if "default" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'default.payment.next.month' or "
            "'default payment next month' target column."
        )
    return df
