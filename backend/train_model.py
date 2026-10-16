from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "demand_forecasting.csv"
MODEL = BASE / "demand_forecasting_model.pkl"
METRICS = BASE / "model_metrics.json"
PREDS = BASE / "test_predictions.csv"
TARGET = "Demand"
CAT = ["Store ID", "Product ID", "Category", "Region"]
NUM = ["Inventory Level", "Price", "Discount", "Promotion", "Competitor Pricing", "Epidemic"]
CAL = ["dayofweek", "month", "dayofyear"]
LAGS = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7"]
FEATURES = CAT + NUM + CAL + LAGS

def add_features(df):
    x = df.copy()
    x["Date"] = pd.to_datetime(x["Date"])
    x = x.sort_values(["Store ID", "Product ID", "Date"]).copy()
    x["dayofweek"] = x["Date"].dt.dayofweek
    x["month"] = x["Date"].dt.month
    x["dayofyear"] = x["Date"].dt.dayofyear
    g = x.groupby(["Store ID", "Product ID"])[TARGET]
    for lag in [1, 7, 14, 28]:
        x[f"lag_{lag}"] = g.shift(lag)
    x["rolling_mean_7"] = x.groupby(["Store ID", "Product ID"])[TARGET].transform(lambda s: s.shift(1).rolling(7).mean())
    return x.sort_values(["Date", "Store ID", "Product ID"]).reset_index(drop=True)

def make_pipeline():
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", SimpleImputer(strategy="median"), NUM + CAL + LAGS)
    ])
    est = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    return Pipeline([("preprocessing", prep), ("model", est)])

def main():
    df = pd.read_csv(DATA)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Store ID", "Product ID"])
    x = add_features(df).dropna(subset=FEATURES + [TARGET]).copy()
    dates = sorted(x["Date"].unique())
    cutoff = dates[int(len(dates) * 0.8)]
    train = x[x["Date"] < cutoff]
    test = x[x["Date"] >= cutoff]
    model = make_pipeline()
    model.fit(train[FEATURES], train[TARGET])
    pred = np.maximum(0, model.predict(test[FEATURES]))
    y = test[TARGET].to_numpy()
    denom = np.abs(y).sum()
    metrics = {
        "model": "Extra Trees Regressor", "test_start": str(test.Date.min().date()),
        "test_end": str(test.Date.max().date()), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "wape_percent": float(np.abs(y-pred).sum()/denom*100) if denom else None,
        "note": "Chronological holdout. This is a rolling one-day-ahead evaluation; covariates must be known at forecast time."
    }
    out = test[["Date", "Store ID", "Product ID", TARGET]].copy()
    out["Predicted_Demand"] = pred
    joblib.dump(model, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    out.to_csv(PREDS, index=False)
    print("\nEXTRA TREES MODEL RESULTS")
    for key in ["mae", "rmse", "r2", "wape_percent"]: print(f"{key}: {metrics[key]:.4f}")
    print(f"Model saved: {MODEL.name}")

if __name__ == "__main__":
    main()
