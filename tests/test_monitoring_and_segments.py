import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import log_prediction


def test_log_prediction_writes_one_row_per_call(tmp_path):
    row = pd.DataFrame([{"LIMIT_BAL": 1000, "AGE": 30, "utilization": 0.5}])
    log_file = str(tmp_path / "logs" / "p.csv")
    log_prediction(row, 0.42, log_file)
    log_prediction(row, 0.10, log_file)
    log = pd.read_csv(log_file)
    assert len(log) == 2
    assert list(log.columns) == ["time", "LIMIT_BAL", "AGE", "utilization", "probability"]
    assert log["probability"].tolist() == [0.42, 0.10]
