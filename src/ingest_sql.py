import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SQL_DIR = os.path.join(BASE_DIR, "sql")
DB_PATH = os.path.join(DATA_DIR, "urban_demand.db")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    with open(os.path.join(SQL_DIR, "schema.sql"), "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    service_df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_service_requests.csv"))
    weather_df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_weather_daily.csv"))
    events_df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_events.csv"))
    holidays_df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_holidays.csv"))

    service_df.to_sql("service_requests", conn, if_exists="replace", index=False)
    weather_df.to_sql("weather_daily", conn, if_exists="replace", index=False)
    events_df.to_sql("city_events", conn, if_exists="replace", index=False)
    holidays_df.to_sql("holidays", conn, if_exists="replace", index=False)

    conn.close()
    print(f"Loaded data into {DB_PATH}")


if __name__ == "__main__":
    main()
