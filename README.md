# Credit Default Risk Prediction (MLOps)

Predicts whether a credit-card client will miss next month's payment. Three models are compared with cross-validation, the best one is tuned, explained with **SHAP**, tracked in **MLflow**, and served through a **FastAPI** endpoint.

**Stack:** Python, pandas, scikit-learn, imbalanced-learn (SMOTE), XGBoost, MLflow, SHAP, FastAPI, Docker, pytest, GitHub Actions

## Business problem
A bank loses far more money when it approves a client who later defaults than when it spends time reviewing a client who would have paid. This project builds a model that flags risky clients early, explains **why** they are risky, groups clients into segments the credit team can act on, and keeps watching the model after it is deployed.

## Files

| File | What it does |
|---|---|
| `features.py` | Builds extra columns (utilization, repayment ratio, delay counts). Used by training AND serving |
| `train.py` | The whole workflow in order: load, features, split, compare models, tune, evaluate, business impact, SHAP, MLflow, save |
| `segments.py` | Groups clients into segments with KMeans (unsupervised) and shows each segment's default rate |
| `api.py` | FastAPI server: `POST /predict` (logs every prediction) |
| `app.py` | Streamlit web demo (deploy this on Hugging Face Spaces) |
| `logger.py`, `monitor.py` | Monitoring: log predictions, then check for data drift |
| `REPORT.md` | One-page business report with recommendations (fill in after your run) |
| `make_sample_data.py` | Makes FAKE data to check that the code runs (never report its results) |

## Data
UCI "Default of Credit Card Clients" (30,000 clients, 23 columns). Download `UCI_Credit_Card.csv` (UCI repository or Kaggle) and save it in `data/`. It is not stored in this repo. The loader also handles CSV exports whose actual column names are on the second row.

## What the training script does
1. **Load and clean** the data; unknown category codes become "other".
2. **Feature engineering:** credit utilization, average bill, total paid, repayment ratio, worst delay, months delayed.
3. **Split** 80/20 with the same default rate in both parts (`stratify`).
4. **Class imbalance:** only ~22% of clients default, so SMOTE creates extra minority examples. It sits **inside** the pipeline so it only touches training folds (no data leakage).
5. **Compare** Logistic Regression, Random Forest and XGBoost with 5-fold cross-validation (ROC-AUC).
6. **Tune** the best model with `GridSearchCV`.
7. **Evaluate** on the untouched test set: ROC-AUC, precision, recall, F1, confusion matrix.
8. **Explain** with a SHAP summary plot (which features push the risk up or down).
9. **Business impact:** converts the confusion matrix into "% of defaulters caught" and an *illustrative* cost reduction. The cost assumptions (a missed default costs 20x a false alarm) are written at the top of that step; change them to match a real bank.
10. **Track and save:** each run is logged in MLflow and the model is saved to `models/model.joblib`.

Then run `python segments.py data/UCI_Credit_Card.csv` for client segments.

## Run it
```bash
pip install -r requirements.txt
python train.py data/UCI_Credit_Card.csv
python segments.py data/UCI_Credit_Card.csv
mlflow ui                                  # compare runs at http://127.0.0.1:5000
streamlit run app.py                       # web demo
uvicorn api:app --reload                   # API: try POST /predict at http://127.0.0.1:8000/docs
python monitor.py                          # after some predictions: drift check
```
Docker (train first so `models/model.joblib` exists): `docker build -t credit-api . && docker run -p 8000:8000 credit-api`

**Live demo:** _paste your Hugging Face Space link here after deploying (see DEPLOY.md at the top of the portfolio)._

## Results (fill in from `results/metrics.json` after your real run)

| Model | CV ROC-AUC |
|---|---|
| Logistic Regression | |
| Random Forest | |
| XGBoost | |

| Test set (threshold 0.5) | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Best model | | | | |

| Business impact (test set) | Value |
|---|---|
| Defaulters caught | |
| Illustrative cost reduction (see assumptions in `train.py`) | |

| Client segment | Clients | Default rate % | What defines it |
|---|---|---|---|
| (from `results/segments.csv`) | | | |

Top features by SHAP (see `results/shap_summary.png`): _write the top 3-5 and what they mean._

## Limitations
- The data is from one Taiwanese bank in 2005, so results may not transfer to other populations.
- A fixed 0.5 threshold is used. A real bank would pick the threshold from the cost of a missed defaulter vs a false alarm.
- `SEX`, `MARRIAGE` and `AGE` are used as in the original dataset; a real credit model needs a fairness and regulatory review.

## Monitoring
Every prediction is saved to `logs/predictions.csv`. `python monitor.py` prints how many predictions were made, how many were flagged, and compares the live average of key inputs with the training average. A change above 20% prints a **drift warning**, which is the signal to retrain.

## Tests
`pytest -q` checks the feature code and the prediction logger.
