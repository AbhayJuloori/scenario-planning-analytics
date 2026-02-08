DROP TABLE IF EXISTS service_requests;
DROP TABLE IF EXISTS weather_daily;
DROP TABLE IF EXISTS city_events;
DROP TABLE IF EXISTS holidays;

CREATE TABLE service_requests (
  date TEXT,
  zone TEXT,
  category TEXT,
  requests INTEGER
);

CREATE TABLE weather_daily (
  date TEXT,
  temp_f REAL,
  precip_in REAL,
  wind_mph REAL
);

CREATE TABLE city_events (
  date TEXT,
  city_event INTEGER
);

CREATE TABLE holidays (
  date TEXT,
  is_holiday INTEGER
);
