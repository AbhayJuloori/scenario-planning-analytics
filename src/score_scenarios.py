import os
import pandas as pd
import numpy as np
from datetime import timedelta
from joblib import load

from feature_engineering import build_features

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_recent_history():
    df = pd.read_csv(os.path.join(REPORTS_DIR, "predictions.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def simulate_future_inputs(last_date, horizon_days=30):
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
    temp = 55 + 20 * np.sin(2 * np.pi * (pd.Series(future_dates).dt.dayofyear / 365.25))
    precip = np.clip(np.random.default_rng(7).gamma(2.0, 0.6, size=horizon_days) - 0.5, 0, None)
    wind = np.clip(np.random.default_rng(8).normal(10, 3, size=horizon_days), 0, None)

    weather = pd.DataFrame(
        {
            "date": future_dates,
            "temp_f": np.round(temp, 1),
            "precip_in": np.round(precip, 2),
            "wind_mph": np.round(wind, 1),
        }
    )

    events = pd.DataFrame({"date": future_dates, "city_event": (np.random.default_rng(9).random(horizon_days) < 0.07).astype(int)})
    holidays = pd.DataFrame({"date": future_dates, "is_holiday": 0})

    return weather, events, holidays


def create_future_frame(history_df, weather_df, events_df, holidays_df):
    latest = history_df.sort_values("date").groupby(["zone", "category"]).tail(14)
    latest["requests"] = latest["prediction"].astype(float)
    latest = latest[["date", "zone", "category", "requests"]]

    future_rows = []
    for (zone, category), group in latest.groupby(["zone", "category"]):
        last_date = group["date"].max()
        future_dates = weather_df["date"].tolist()
        for d in future_dates:
            future_rows.append(
                {
                    "date": d,
                    "zone": zone,
                    "category": category,
                    "requests": np.nan,
                }
            )

    future_df = pd.DataFrame(future_rows)

    combined = pd.concat([latest, future_df], ignore_index=True)
    combined["is_future"] = combined["requests"].isna().astype(int)
    combined = combined.merge(weather_df, on="date", how="left")
    combined = combined.merge(events_df, on="date", how="left")
    combined = combined.merge(holidays_df, on="date", how="left")

    combined["city_event"] = combined["city_event"].fillna(0)
    combined["is_holiday"] = combined["is_holiday"].fillna(0)

    combined = combined.sort_values(["zone", "category", "date"]).reset_index(drop=True)
    # Forward-fill requests so lag/rolling features exist for the whole horizon
    combined["requests"] = (
        combined.groupby(["zone", "category"])["requests"].ffill().astype(float)
    )
    return combined


def compute_scenarios(forecast_df):
    # Base capacity per day
    base_capacity = 110
    surge_capacity = 135
    constrained_capacity = 95
    overtime_rate = 1.5

    rows = []
    for scenario, capacity in [
        ("base", base_capacity),
        ("surge", surge_capacity),
        ("constrained", constrained_capacity),
    ]:
        grouped = forecast_df.groupby("date")["prediction"].sum().reset_index()
        grouped["capacity"] = capacity * forecast_df["zone"].nunique()
        grouped["service_level"] = (grouped["capacity"] / grouped["prediction"]).clip(0, 1)
        grouped["overtime_units"] = (grouped["prediction"] - grouped["capacity"]).clip(lower=0)
        grouped["overtime_cost_index"] = grouped["overtime_units"] * overtime_rate
        grouped["scenario"] = scenario
        rows.append(grouped)

    result = pd.concat(rows, ignore_index=True)
    return result


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    bundle = load(os.path.join(MODELS_DIR, "model_bundle.joblib"))
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    history_df = load_recent_history()
    last_date = history_df["date"].max()

    weather_df, events_df, holidays_df = simulate_future_inputs(last_date, horizon_days=30)
    combined = create_future_frame(history_df, weather_df, events_df, holidays_df)

    combined, _ = build_features(combined, dropna_requests=False)

    future_df = combined[combined["is_future"] == 1].copy()
    X_future = future_df[feature_cols]
    future_df["prediction"] = model.predict(X_future)

    scenario_df = compute_scenarios(future_df)
    scenario_df.to_csv(os.path.join(REPORTS_DIR, "scenario_results.csv"), index=False)

    print("Scenario results written to reports/")


if __name__ == "__main__":
    main()
