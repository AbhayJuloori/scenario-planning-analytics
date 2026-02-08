# Urban Demand Planning & Scenario Analytics Pipeline

## What This Project Is (Simple Explanation)
Cities and organizations get lots of requests every day (trash pickup issues, broken street lights, potholes, public safety calls, etc.). If they can **predict how many requests will come in**, they can plan staff, budget, and equipment better.

This project builds a **prediction + planning pipeline** that:
- learns patterns from past data
- forecasts future demand
- turns those forecasts into **planning scenarios** (normal, surge, and constrained)
- shows results in a simple dashboard

Everything here uses **synthetic data** that looks realistic but is safe to publish.

---

## Why This Project Is Needed
- **Better planning**: Cities and teams don’t want to guess staffing levels.
- **Cost control**: Forecasts help avoid sudden overtime costs.
- **Service quality**: If demand is higher than capacity, service levels drop.

This project shows how data science can directly support real decisions — not just accuracy metrics.

---

## How It Works (Step by Step)
1. **Generate data** (fake but realistic patterns)
2. **Store it in SQL** (like real analytics pipelines)
3. **Create features** (weather, holidays, trends, and past demand)
4. **Train a model** (predict daily demand)
5. **Evaluate the model** (MAE, RMSE, MAPE, R²)
6. **Create planning scenarios** (base / surge / constrained)
7. **Show results in a dashboard**

---

## Data Used
The pipeline creates four tables (CSV + SQLite):
- `service_requests`: daily request counts by zone and category
- `weather_daily`: temperature, rain, wind
- `city_events`: event days (more demand)
- `holidays`: holiday indicator

These are **synthetic** but shaped like real public datasets (311, bike‑sharing, etc.).

---

## Outputs (What You Get)
After running the pipeline, you get:

### 1. Model Metrics
File: `reports/model_metrics.csv`
- **MAE**: average error
- **RMSE**: bigger penalties for large errors
- **MAPE**: average percent error
- **R²**: how much variance the model explains

### 2. Predictions
File: `reports/predictions.csv`
- predicted request counts for each zone + category

### 3. Scenario Planning Results
File: `reports/scenario_results.csv`
For each day and scenario:
- **service_level** = capacity / demand (higher = better)
- **overtime_cost_index** = how much overtime is needed

### 4. Dashboard
Run Streamlit to see charts:
- Service level over time (base / surge / constrained)
- Overtime cost index over time

---

## How to Run
```bash
cd urban-demand-planning-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/generate_data.py
python src/ingest_sql.py
python src/train_model.py
python src/score_scenarios.py

streamlit run dashboard/app_streamlit.py
```

---

## Project Structure
```
urban-demand-planning-pipeline/
  src/
    generate_data.py
    ingest_sql.py
    feature_engineering.py
    train_model.py
    score_scenarios.py
  sql/
    schema.sql
    queries.sql
  data/               # generated after running
  reports/            # generated after running
  models/             # generated after running
  dashboard/
    app_streamlit.py
  requirements.txt
```

---

## Tech Stack
- Python
- Pandas + NumPy
- Scikit‑learn
- SQLite
- Streamlit + Plotly

---

## Notes
- Data is synthetic (safe for GitHub).
- You can replace the data with a real dataset if desired.
- This project is built to show **end‑to‑end analytics**, not just modeling.
