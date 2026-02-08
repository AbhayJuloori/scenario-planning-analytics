import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
import holidays

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RNG = np.random.default_rng(42)

ZONES = ["North", "South", "East", "West", "Central", "Harbor", "Airport", "Hills"]
CATEGORIES = ["Sanitation", "Roads", "Utilities", "PublicSafety"]

START_DATE = date(2023, 1, 1)
END_DATE = date(2024, 12, 31)


def daterange(start, end):
    curr = start
    while curr <= end:
        yield curr
        curr += timedelta(days=1)


def create_weather(days):
    # Simple seasonal temperature curve + noise
    day_of_year = np.array([d.timetuple().tm_yday for d in days])
    temp = 55 + 20 * np.sin(2 * np.pi * (day_of_year / 365.25))
    temp += RNG.normal(0, 6, size=len(days))
    precip = np.clip(RNG.gamma(2.0, 0.6, size=len(days)) - 0.5, 0, None)
    wind = np.clip(RNG.normal(10, 3, size=len(days)), 0, None)
    return pd.DataFrame(
        {
            "date": days,
            "temp_f": np.round(temp, 1),
            "precip_in": np.round(precip, 2),
            "wind_mph": np.round(wind, 1),
        }
    )


def create_events(days):
    event_prob = 0.06
    events = RNG.random(len(days)) < event_prob
    return pd.DataFrame({"date": days, "city_event": events.astype(int)})


def create_holidays(days):
    us_holidays = holidays.UnitedStates(years=[2023, 2024])
    flags = [1 if d in us_holidays else 0 for d in days]
    return pd.DataFrame({"date": days, "is_holiday": flags})


def create_service_requests(days, weather_df, events_df, holidays_df):
    rows = []
    weather_df = weather_df.set_index("date")
    events_df = events_df.set_index("date")
    holidays_df = holidays_df.set_index("date")

    for d in days:
        dow = d.weekday()
        is_weekend = 1 if dow >= 5 else 0
        event = events_df.loc[d, "city_event"]
        holiday = holidays_df.loc[d, "is_holiday"]
        temp = weather_df.loc[d, "temp_f"]
        precip = weather_df.loc[d, "precip_in"]

        for zone in ZONES:
            zone_factor = 0.85 + 0.3 * (ZONES.index(zone) / (len(ZONES) - 1))
            for cat in CATEGORIES:
                base = 40 + 10 * CATEGORIES.index(cat)
                seasonal = 8 * np.sin(2 * np.pi * (d.timetuple().tm_yday / 365.25))
                weekend_effect = -10 if is_weekend else 5
                event_effect = 12 if event else 0
                holiday_effect = -8 if holiday else 0
                weather_effect = -0.2 * (temp - 60) + 2.5 * precip

                mean = (
                    base
                    + seasonal
                    + weekend_effect
                    + event_effect
                    + holiday_effect
                    + weather_effect
                )
                mean = max(5, mean * zone_factor)
                count = RNG.poisson(lam=mean)

                rows.append(
                    {
                        "date": d,
                        "zone": zone,
                        "category": cat,
                        "requests": int(count),
                    }
                )

    return pd.DataFrame(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    days = list(daterange(START_DATE, END_DATE))

    weather_df = create_weather(days)
    events_df = create_events(days)
    holidays_df = create_holidays(days)
    requests_df = create_service_requests(days, weather_df, events_df, holidays_df)

    weather_df.to_csv(os.path.join(DATA_DIR, "synthetic_weather_daily.csv"), index=False)
    events_df.to_csv(os.path.join(DATA_DIR, "synthetic_events.csv"), index=False)
    holidays_df.to_csv(os.path.join(DATA_DIR, "synthetic_holidays.csv"), index=False)
    requests_df.to_csv(
        os.path.join(DATA_DIR, "synthetic_service_requests.csv"), index=False
    )

    print("Synthetic data generated in data/")


if __name__ == "__main__":
    main()
