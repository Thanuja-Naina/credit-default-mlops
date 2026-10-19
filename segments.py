"""Customer segmentation with KMeans (UNSUPERVISED learning: no target column is used for grouping).

Idea: group clients with similar behaviour, THEN look at each group's default rate.
This tells a bank which kinds of clients are risky, in plain business language.

Run: python segments.py data/UCI_Credit_Card.csv
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_utils import load_credit_data
from features import add_features

path = sys.argv[1] if len(sys.argv) > 1 else "data/UCI_Credit_Card.csv"
df = load_credit_data(path)
df = add_features(df)

# 1) Choose the behaviour columns and put them on the same scale
columns = ["LIMIT_BAL", "AGE", "utilization", "repay_ratio", "max_delay", "months_delayed"]
X = StandardScaler().fit_transform(df[columns])

# 2) Pick the number of groups (k) with the silhouette score (higher = better separated groups)
rng = np.random.default_rng(42)
sample = X[rng.choice(len(X), min(5000, len(X)), replace=False)]    # a sample keeps this fast
silhouettes = {k: silhouette_score(sample, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(sample))
               for k in range(2, 7)}
best_k = max(silhouettes, key=silhouettes.get)
print("Silhouette scores:", {k: round(v, 3) for k, v in silhouettes.items()}, "-> using k =", best_k)

# 3) Fit KMeans and describe each segment
df["segment"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X)
summary = df.groupby("segment").agg(
    clients=("default", "size"),
    default_rate_pct=("default", lambda s: round(s.mean() * 100, 1)),
    avg_limit=("LIMIT_BAL", "mean"),
    avg_utilization=("utilization", "mean"),
    avg_max_delay=("max_delay", "mean"),
).round(2)
print(summary)

summary.to_csv("results/segments.csv")
summary["default_rate_pct"].plot.bar(title="Default rate by client segment (%)", xlabel="segment")
plt.savefig("results/segments_default_rate.png", dpi=120, bbox_inches="tight")
print("Saved results/segments.csv and results/segments_default_rate.png")
