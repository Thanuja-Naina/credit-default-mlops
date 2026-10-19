import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import load_credit_data


def test_load_credit_data_handles_second_row_headers(tmp_path):
    path = tmp_path / "credit.csv"
    path.write_text(
        "Unnamed: 0,X1,X2,Y\n"
        "ID,LIMIT_BAL,AGE,default payment next month\n"
        "1,20000,24,1\n",
        encoding="utf-8",
    )

    data = load_credit_data(path)

    assert data.columns.tolist() == ["ID", "LIMIT_BAL", "AGE", "default"]
    assert data.loc[0, "default"] == 1
