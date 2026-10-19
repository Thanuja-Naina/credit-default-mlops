"""MONITORING helper: save every prediction to a CSV file so we can study it later (see monitor.py)."""
import csv
import os
from datetime import datetime

LOG_FILE = "logs/predictions.csv"


def log_prediction(row, probability, log_file=LOG_FILE):
    """row = the one-row DataFrame after add_features(); probability = the model's default risk."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    is_new_file = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["time", "LIMIT_BAL", "AGE", "utilization", "probability"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), row["LIMIT_BAL"].iloc[0], row["AGE"].iloc[0],
                         round(row["utilization"].iloc[0], 4), round(probability, 4)])
