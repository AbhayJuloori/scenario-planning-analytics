import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump

from feature_engineering import build_features

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "urban_demand.db")


def load_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        sr.date,
        sr.zone,
        sr.category,
        sr.requests,
        w.temp_f,
        w.precip_in,
        w.wind_mph,
        e.city_event,
        h.is_holiday
    FROM service_requests sr
    JOIN weather_daily w ON sr.date = w.date
    JOIN city_events e ON sr.date = e.date
    JOIN holidays h ON sr.date = h.date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def train_test_split_time(df, cutoff_date):
    train = df[df["date"] < cutoff_date].copy()
    test = df[df["date"] >= cutoff_date].copy()
    return train, test


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data()
    df, feature_cols = build_features(df)

    df["date"] = pd.to_datetime(df["date"])
    cutoff_date = pd.Timestamp("2024-09-01")
    train_df, test_df = train_test_split_time(df, cutoff_date)

    X_train = train_df[feature_cols]
    y_train = train_df["requests"]
    X_test = test_df[feature_cols]
    y_test = test_df["requests"]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae, rmse, mape, r2 = evaluate(y_test, preds)

    metrics_df = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "MAPE", "R2"],
            "value": [mae, rmse, mape, r2],
        }
    )
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "model_metrics.csv"), index=False)

    output_df = test_df[["date", "zone", "category", "requests"]].copy()
    output_df["prediction"] = np.round(preds, 1)
    output_df.to_csv(os.path.join(REPORTS_DIR, "predictions.csv"), index=False)

    dump(
        {"model": model, "feature_cols": feature_cols},
        os.path.join(MODELS_DIR, "model_bundle.joblib"),
    )

    print("Model trained. Metrics and predictions written to reports/")


if __name__ == "__main__":
    main()
