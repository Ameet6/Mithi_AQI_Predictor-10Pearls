"""
Prediction helpers.
Loads the active model for a given horizon from the MongoDB model registry,
and generates predictions using CURRENT pollutant levels combined with
FORECASTED weather for the target day (not today's weather) — matching
how the models were trained (see train.py's build_target_and_split).
"""

import io
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd

from src import config, db
from src.features.fetch import fetch_forecast_weather

HORIZONS_HOURS = [24, 48, 72]


def get_latest_features(collection) -> dict:
    """Fetch the single most recent feature document for our city."""
    doc = collection.find_one({"city": config.CITY_NAME}, sort=[("timestamp", -1)])
    return doc


def load_active_model(horizon_hours: int, models_collection) -> dict:
    """Fetch the currently active model + its metadata for a given horizon."""
    doc = models_collection.find_one({
        "city": config.CITY_NAME,
        "horizon_hours": horizon_hours,
        "is_active": True,
    })
    if doc is None:
        return None

    if doc["algorithm"] == "xgboost":
        import tempfile, os
        from xgboost import XGBRegressor
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(doc["model_binary"])
            tmp_path = tmp.name
        model = XGBRegressor()
        model.load_model(tmp_path)
        os.remove(tmp_path)
    else:
        model = joblib.load(io.BytesIO(doc["model_binary"]))
    return {
        "model": model,
        "algorithm": doc["algorithm"],
        "feature_cols": doc["feature_cols"],
        "metrics": doc["metrics"],
        "trained_at": doc["trained_at"],
    }


def get_forecast_weather_for_target(forecast_hourly: dict, target_time: datetime) -> dict:
    """
    Given Open-Meteo's hourly forecast arrays and a target datetime, find
    the hourly entry closest to that target time and return its weather
    values in the same shape used during training.
    """
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in forecast_hourly["time"]]

    closest_idx = min(range(len(times)), key=lambda i: abs((times[i] - target_time).total_seconds()))

    return {
        "temperature": forecast_hourly["temperature_2m"][closest_idx],
        "humidity": forecast_hourly["relative_humidity_2m"][closest_idx],
        "pressure": forecast_hourly["surface_pressure"][closest_idx],
        "wind_speed": forecast_hourly["wind_speed_10m"][closest_idx],
    }


def predict_all_horizons(latest_features: dict, models_collection) -> list:
    """
    Run all 3 horizon models using:
    - current pollutant levels (from latest_features)
    - forecasted weather for each target day (fetched live from Open-Meteo)
    - calendar features computed directly from the target timestamp
    """
    now = latest_features["timestamp"]
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    forecast_hourly = fetch_forecast_weather(config.LAT, config.LON, forecast_days=4)

    results = []
    for horizon_hours in HORIZONS_HOURS:
        model_info = load_active_model(horizon_hours, models_collection)
        if model_info is None:
            results.append({
                "horizon_hours": horizon_hours, "day": horizon_hours // 24,
                "predicted_aqi": None, "algorithm": None, "metrics": None,
            })
            continue

        target_time = now + timedelta(hours=horizon_hours)
        weather = get_forecast_weather_for_target(forecast_hourly, target_time)

        feature_row = {
            "aqi": latest_features["aqi"],
            "pm2_5": latest_features["pm2_5"],
            "pm10": latest_features["pm10"],
            "co": latest_features["co"],
            "no2": latest_features["no2"],
            "so2": latest_features["so2"],
            "o3": latest_features["o3"],
            "aqi_change_rate": latest_features["aqi_change_rate"],
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "pressure": weather["pressure"],
            "wind_speed": weather["wind_speed"],
            "hour": target_time.hour,
            "day": target_time.day,
            "month": target_time.month,
            "day_of_week": target_time.weekday(),
        }

        X = pd.DataFrame([{col: feature_row.get(col) for col in model_info["feature_cols"]}])
        prediction = model_info["model"].predict(X)[0]

        results.append({
            "horizon_hours": horizon_hours,
            "day": horizon_hours // 24,
            "predicted_aqi": round(float(prediction), 1),
            "algorithm": model_info["algorithm"],
            "metrics": model_info["metrics"],
        })

    return results


def aqi_category(aqi_value: float) -> tuple:
    """
    Map a US AQI value to its official health category, a display color,
    and a matching text color that stays legible on that background.
    Returns (label, bg_color, text_color).
    """
    if aqi_value is None:
        return ("Unknown", "#94a3b8", "#ffffff")
    if aqi_value <= 50:
        return ("Good", "#16a34a", "#ffffff")
    if aqi_value <= 100:
        return ("Moderate", "#eab308", "#1a2332")
    if aqi_value <= 150:
        return ("Unhealthy for Sensitive Groups", "#f97316", "#ffffff")
    if aqi_value <= 200:
        return ("Unhealthy", "#dc2626", "#ffffff")
    if aqi_value <= 300:
        return ("Very Unhealthy", "#9333ea", "#ffffff")
    return ("Hazardous", "#7f1d1d", "#ffffff")
HEALTH_GUIDANCE = {
    "Good": "Air quality is satisfactory. Enjoy normal outdoor activities.",
    "Moderate": "Acceptable air quality. Unusually sensitive individuals should consider limiting prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Children, elderly, and those with asthma or heart conditions should limit prolonged outdoor exertion. Consider wearing a mask outdoors.",
    "Unhealthy": "Everyone should limit prolonged outdoor exertion. Sensitive groups should avoid outdoor activity entirely. Wear an N95 mask if going outside.",
    "Very Unhealthy": "Avoid outdoor activity. Keep windows closed. Sensitive groups should remain indoors with air filtration if possible.",
    "Hazardous": "Stay indoors. Avoid all outdoor exertion. Use air purifiers indoors if available. Seek medical attention if experiencing symptoms.",
    "Unknown": "No guidance available.",
}


def get_health_guidance(label: str) -> str:
    """Return actionable health guidance text for a given AQI category label."""
    return HEALTH_GUIDANCE.get(label, HEALTH_GUIDANCE["Unknown"])