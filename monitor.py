"""MONITORING: read the prediction log and check the model is still behaving normally.  Run: python monitor.py

Data drift = the clients we see today look different from the clients the model was trained on.
If that happens, the model's accuracy may drop and it is time to retrain.
"""
import json
import os

import pandas as pd

if not os.path.exists("logs/predictions.csv"):
    raise SystemExit("No predictions logged yet. Call the API (api.py or app.py) first.")

log = pd.read_csv("logs/predictions.csv")
training_stats = json.load(open("models/training_stats.json"))

print(f"Predictions logged     : {len(log)}")
print(f"Share flagged as risky : {(log['probability'] >= 0.5).mean():.1%}")
print(f"Average default risk   : {log['probability'].mean():.3f}")

print("\nDrift check (live average vs training average):")
for column in ["LIMIT_BAL", "AGE", "utilization"]:
    change = (log[column].mean() - training_stats[column]) / abs(training_stats[column]) * 100
    warning = "  <-- WARNING: possible drift, consider retraining" if abs(change) > 20 else ""
    print(f"  {column:12s} training = {training_stats[column]:>10.2f} | live = {log[column].mean():>10.2f} | change = {change:+.1f}%{warning}")
