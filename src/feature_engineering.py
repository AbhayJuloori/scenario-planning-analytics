import pandas as pd


def build_features(df, dropna_requests=True):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.weekday
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df = df.sort_values(["zone", "category", "date"]).reset_index(drop=True)
    df["lag_1"] = df.groupby(["zone", "category"])["requests"].shift(1)
    df["lag_7"] = df.groupby(["zone", "category"])["requests"].shift(7)
    df["rolling_7"] = (
        df.groupby(["zone", "category"])["requests"]
        .apply(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
        .reset_index(level=[0, 1], drop=True)
    )

    if dropna_requests:
        df = df.dropna().reset_index(drop=True)
    else:
        df = df.dropna(
            subset=[
                "temp_f",
                "precip_in",
                "wind_mph",
                "city_event",
                "is_holiday",
                "lag_1",
                "lag_7",
                "rolling_7",
            ]
        ).reset_index(drop=True)

    feature_cols = [
        "temp_f",
        "precip_in",
        "wind_mph",
        "city_event",
        "is_holiday",
        "dow",
        "weekofyear",
        "month",
        "is_weekend",
        "lag_1",
        "lag_7",
        "rolling_7",
    ]

    return df, feature_cols
