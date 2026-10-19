import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import NEW_COLS, RAW_COLS, add_features


def make_row():
    row = {c: 1 for c in RAW_COLS}
    row.update({"LIMIT_BAL": 1000, "BILL_AMT1": 500, "EDUCATION": 6, "MARRIAGE": 0})
    return pd.DataFrame([row])


def test_new_columns_are_created():
    out = add_features(make_row())
    assert all(c in out.columns for c in NEW_COLS)
    assert out.loc[0, "utilization"] == 0.5


def test_unknown_category_codes_become_other():
    out = add_features(make_row())
    assert out.loc[0, "EDUCATION"] == 4 and out.loc[0, "MARRIAGE"] == 3
