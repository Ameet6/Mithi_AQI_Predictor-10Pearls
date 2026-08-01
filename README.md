# Mithi AQI Predictor

An end-to-end, serverless Air Quality Index (AQI) forecasting system for **Mithi, Sindh, Pakistan** — a small town in the Thar Desert. The system collects live weather and pollution data hourly, retrains its forecasting models daily, and serves a 3-day AQI forecast through an interactive, explainable dashboard.

**Live dashboard:** [https://mithiaqipredictor-10pearls.streamlit.app/]

---

## Project Overview

Given Mithi's coordinates, this system predicts AQI for the next 3 days (24h / 48h / 72h ahead), using a fully serverless stack — no servers to manage, no infrastructure to maintain. Data collection, model training, and the dashboard all run on free-tier cloud services, automated end to end.

## Architecture

```
Open-Meteo API (weather + air quality)
        │
        ▼
Feature Pipeline (hourly, GitHub Actions)
        │
        ▼
MongoDB Atlas — feature store ("features" collection)
        │
        ▼
Training Pipeline (daily, GitHub Actions)
        │
        ▼
MongoDB Atlas — model registry ("models" collection, binary-stored)
        │
        ▼
Streamlit Dashboard (live, Streamlit Community Cloud)
```

## Key Design Decisions

**Data source: Open-Meteo, not a paid weather API.** Open-Meteo provides both historical and forecasted weather *and* air quality data, entirely free with no API key required — this avoided vendor lock-in and paid-tier limitations common with alternatives.

**Feature store: MongoDB Atlas, not a dedicated feature-store product.** Every hourly reading is stored as one document, deduplicated on `(city, hour_bucket)` via upsert, so re-running the pipeline never creates duplicates. This is a lightweight but effective substitute for a managed feature store, chosen to keep the stack simple and free.

**Model registry: models stored as binary directly in MongoDB, not on disk.** Since GitHub Actions runners are ephemeral (torn down after each run) and the dashboard runs in a separate environment (Streamlit Cloud), a locally-saved model file would be inaccessible from anywhere except the machine that trained it. Storing the serialized model as binary inside the same MongoDB database means training (wherever it runs) and prediction (wherever it runs) always share one source of truth.

**Forecast-based feature engineering, not just "today's conditions."** Early iterations of this project predicted 3-day-ahead AQI using only the current day's weather and pollution readings as inputs. This performed poorly (R² well below 0.5 at every horizon) because today's weather has limited predictive power for weather 3 days out. The final design instead uses:
- **Pollutant levels from now** (the most recent real information available)
- **Forecasted weather for the target day** (fetched live from Open-Meteo's forecast API at prediction time; approximated during training using the actual historical weather that occurred, since a 1-3 day forecast is typically close to what actually happens)
- **Calendar features for the target day** (hour, day, month, day of week — always exactly knowable in advance, regardless of horizon)

This redesign substantially improved forecast accuracy (see Results below).

**Multi-horizon models, not one single model.** Rather than one model trying to average performance across "the next 3 days" broadly, a separate model is trained and evaluated independently for each of Day 1 (24h), Day 2 (48h), and Day 3 (72h) — with the best-performing algorithm selected per horizon, since accuracy naturally degrades with distance and different algorithms suit different horizons.

## Exploratory Data Analysis — Key Findings

- **Strong seasonal pattern**: AQI in Mithi peaks in the pre-monsoon months (June–July), driven by dust storms, and drops sharply once monsoon rains arrive (August–September) — the opposite seasonal shape of a typical urban-pollution-driven city, since Mithi's air quality is dominated by desert dust rather than traffic/industrial emissions.
- **`pm2_5` is the strongest single correlate with AQI** (0.76), though extreme AQI events are driven by `pm10` (coarse dust particles) — confirmed by inspecting the top 10 highest-AQI hours, which cluster on a single date with PM10 far exceeding PM2.5, a clear dust-storm signature.
- **Wind speed positively correlates with pollution** (0.17–0.47 depending on the pollutant) — the opposite relationship you'd see in an urban setting, since wind here kicks up dust rather than dispersing it.
- Extreme AQI values were validated as genuine multi-hour pollution events (not sensor errors) by confirming they cluster in time and that multiple pollutants move together consistently.

## Modeling

Five candidate algorithms were trained and compared at each forecast horizon: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, and XGBoost. The best performer was selected independently per horizon based on R².

### Results

| Horizon | Best Model | RMSE | MAE | R² |
|---|---|---|---|---|
| Day 1 (24h) | Linear Regression | 13.04 | 9.13 | **0.718** |
| Day 2 (48h) | Random Forest | 18.94 | 12.28 | **0.407** |
| Day 3 (72h) | XGBoost | 21.43 | 14.51 | **0.240** |

Accuracy degrades with forecast distance, as expected — Day 1 benefits from highly accurate near-term weather forecasts and strong AQI persistence, while Day 3 faces greater genuine uncertainty in both weather and pollution dynamics. This is a well-known, industry-wide limitation of multi-day air quality forecasting, not unique to this system.

Interestingly, the best-performing model type changes by horizon: **linear regression wins at Day 1**, where the forecast-weather-to-AQI relationship is close to direct and near-term forecasts are highly accurate, while **tree-based models (Random Forest, XGBoost) win at Days 2-3**, where relationships become noisier and less linear.

### Train/test methodology

Since this is time-series data, evaluation uses a **chronological split** (85% earliest data for training, 15% most recent for testing) rather than a random split — this correctly simulates real deployment, where the model only ever has access to the past when predicting the future. Feature/target pairs are aligned by exact timestamp arithmetic (not row position) to guard against any gaps in hourly data silently corrupting the training set.

## Explainability

Each forecast includes a live SHAP (SHapley Additive exPlanations) breakdown showing which features pushed that specific prediction up or down, and by how much. This works across all model types used (linear and tree-based) via SHAP's unified `Explainer` API with a background sample for baseline comparison.

## Automation (CI/CD)

- **Feature pipeline**: runs hourly via GitHub Actions, fetching live conditions and appending to the feature store
- **Training pipeline**: runs daily via GitHub Actions, retraining all 3 horizon models on the latest data and updating the active model registry
- **Model retention**: each training run keeps only the 3 most recent versions per horizon, automatically deleting older ones to prevent unbounded database growth over time

## Alerts

The dashboard displays a two-tier hazardous AQI alert banner, checking both current conditions and all 3 forecasted days:
- **Caution** (AQI ≥ 100): sensitive groups advised to limit prolonged outdoor exertion
- **Warning** (AQI ≥ 150): general public advised to limit outdoor activity

Alert messaging and in-panel health guidance are generated from a single shared source, ensuring consistency across the dashboard.

## Tech Stack

| Component | Technology |
|---|---|
| Data source | Open-Meteo (Weather + Air Quality APIs) |
| Feature store / model registry | MongoDB Atlas (free tier) |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| Automation | GitHub Actions |
| Dashboard | Streamlit, Plotly |
| Hosting | Streamlit Community Cloud |

## Project Structure

```
Mithi_AQI_Predictor/
├── src/
│   ├── config.py              # City/region config, env vars
│   ├── db.py                  # Shared MongoDB connection
│   ├── alerts.py               # Hazardous AQI alert logic
│   ├── explain.py              # SHAP explanation logic
│   ├── predict.py              # Live prediction logic
│   ├── train.py                # Multi-horizon training pipeline
│   ├── backfill.py             # Historical data backfill
│   └── features/
│       ├── fetch.py            # Open-Meteo API calls (current, historical, forecast)
│       ├── engineer.py         # Feature engineering logic
│       └── pipeline.py         # Live feature pipeline orchestration
├── app/
│   └── dashboard.py            # Streamlit dashboard
├── notebooks/
│   ├── eda.ipynb                # Exploratory data analysis
│   └── model_experiments.ipynb  # Model comparison experiments
├── .github/workflows/
│   ├── feature_pipeline.yml     # Hourly automation
│   └── training_pipeline.yml    # Daily automation
└── requirements.txt
```

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`, fill in your MongoDB Atlas connection string
3. Run the feature pipeline once: `python -m src.features.pipeline`
4. Backfill historical data: `python -m src.backfill`
5. Train models: `python -m src.train`
6. Launch the dashboard: `streamlit run app/dashboard.py`

## Limitations & Honest Notes

- Training uses actual historical weather as a stand-in for forecasted weather at the target time (a standard, disclosed approximation in forecasting systems); live predictions use genuine forecast data, which carries some forecast error not present during training — real-world accuracy may be marginally lower than backtested numbers, particularly at longer horizons.
- Day 3 (72h) accuracy remains modest (R² 0.24) — this reflects the genuine difficulty of multi-day air quality forecasting industry-wide, not a specific flaw in this implementation.
- `nh3` (ammonia) is excluded from features — Open-Meteo's air quality model does not report this pollutant for this location.
