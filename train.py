"""Train, compare, tune, evaluate and explain a credit-default model.

Run with the real data:   python train.py data/UCI_Credit_Card.csv
Quick test with fake data: python make_sample_data.py && python train.py data/sample_fake_data.csv
"""
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")                       # draw plots to files, no window needed
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline      # like sklearn's Pipeline, but allows SMOTE as a step
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data_utils import load_credit_data
from features import CATEGORY_COLS, NUMBER_COLS, RAW_COLS, add_features

# ---------------------------------------------------------------- 1. Load data
path = sys.argv[1] if len(sys.argv) > 1 else "data/UCI_Credit_Card.csv"
df = load_credit_data(path)
df = df.drop(columns=["ID"], errors="ignore")
print(f"Rows: {len(df)} | default rate: {df['default'].mean():.2%}")

# ---------------------------------------------------------------- 2. Features + split
df = add_features(df)
X = df[RAW_COLS + ["utilization", "avg_bill", "total_paid", "repay_ratio", "max_delay", "months_delayed"]]
y = df["default"]
# stratify keeps the same default rate in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------------------------------------------- 3. Preprocessing
# scale the numbers, one-hot encode the categories
prep = ColumnTransformer([
    ("numbers", StandardScaler(), NUMBER_COLS),
    ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORY_COLS),
])


def make_pipeline(model):
    # SMOTE (fixing class imbalance) is INSIDE the pipeline so it only touches training folds.
    # This avoids "data leakage" - the validation data never gets synthetic samples.
    return Pipeline([("prep", prep), ("smote", SMOTE(random_state=42)), ("model", model)])


# ---------------------------------------------------------------- 4. Compare 3 models with cross-validation
models = {
    "logreg": LogisticRegression(max_iter=2000),
    "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
}
mlflow.set_experiment("credit-default")
scores = {}
for name, model in models.items():
    cv_scores = cross_val_score(make_pipeline(model), X_train, y_train, cv=5, scoring="roc_auc")
    scores[name] = cv_scores.mean()
    print(f"{name:14s} CV ROC-AUC = {scores[name]:.4f}")
    with mlflow.start_run(run_name=f"cv-{name}"):
        mlflow.log_param("model", name)
        mlflow.log_metric("cv_roc_auc", scores[name])

best_name = max(scores, key=scores.get)
print("Best model:", best_name)

# ---------------------------------------------------------------- 5. Tune the best model (GridSearchCV)
grids = {
    "logreg": {"model__C": [0.1, 1, 10]},
    "random_forest": {"model__max_depth": [8, 14, None], "model__min_samples_leaf": [1, 3, 5]},
    "xgboost": {"model__max_depth": [3, 4, 6], "model__learning_rate": [0.03, 0.05, 0.1]},
}
search = GridSearchCV(make_pipeline(models[best_name]), grids[best_name], scoring="roc_auc", cv=5)
search.fit(X_train, y_train)
best_pipeline = search.best_estimator_
print("Best settings:", search.best_params_)

# ---------------------------------------------------------------- 6. Evaluate on the untouched test set
probabilities = best_pipeline.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)
metrics = {
    "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
    "precision": round(precision_score(y_test, predictions), 4),
    "recall": round(recall_score(y_test, predictions), 4),   # recall = share of real defaulters we caught
    "f1": round(f1_score(y_test, predictions), 4),
}
print("Test metrics:", metrics)

# ---------------------------------------------------------------- 6b. Business impact
# ILLUSTRATIVE assumptions - change them to match a real bank. Costs can be in any money unit.
COST_MISSED_DEFAULT = 10000    # we approve a client who then defaults
COST_FALSE_ALARM = 500         # we manually review a client who would have paid
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
cost_without_model = (tp + fn) * COST_MISSED_DEFAULT                  # no model: every defaulter is missed
cost_with_model = fn * COST_MISSED_DEFAULT + fp * COST_FALSE_ALARM
metrics["defaulters_caught_pct"] = round(tp / (tp + fn) * 100, 1)
metrics["illustrative_cost_reduction_pct"] = round((1 - cost_with_model / cost_without_model) * 100, 1)
print(f"Business impact: the model catches {metrics['defaulters_caught_pct']}% of defaulters. "
      f"Illustrative cost reduction: {metrics['illustrative_cost_reduction_pct']}% "
      f"(assumes a missed default costs {COST_MISSED_DEFAULT // COST_FALSE_ALARM}x a false alarm)")

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Confusion matrix")
plt.savefig("results/confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- 7. Explain the model with SHAP
X_sample = best_pipeline.named_steps["prep"].transform(X_test.sample(min(300, len(X_test)), random_state=0))
feature_names = best_pipeline.named_steps["prep"].get_feature_names_out()
explanation = shap.Explainer(best_pipeline.named_steps["model"], X_sample)(X_sample)
values = explanation.values[:, :, 1] if explanation.values.ndim == 3 else explanation.values
shap.summary_plot(values, X_sample, feature_names=feature_names, show=False)
plt.savefig("results/shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- 8. Log to MLflow and save the model
with mlflow.start_run(run_name=f"final-{best_name}"):
    mlflow.log_param("model", best_name)
    mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("results/confusion_matrix.png")
    mlflow.log_artifact("results/shap_summary.png")

joblib.dump(best_pipeline, "models/model.joblib")
# average values of the training data, used by monitor.py to detect data drift later
training_stats = X_train[["LIMIT_BAL", "AGE", "utilization"]].mean().to_dict()
json.dump(training_stats, open("models/training_stats.json", "w"), indent=2)
json.dump({"model": best_name, "test_metrics": metrics, "data": path}, open("results/metrics.json", "w"), indent=2)
print("Saved models/model.joblib, models/training_stats.json and results/metrics.json")
