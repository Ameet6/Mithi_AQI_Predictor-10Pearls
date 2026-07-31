"""
Pearls AQI Predictor — Training Pipeline (multi-horizon)
------------------------------------------------------------
For each forecast horizon (24h, 48h, 72h = Day 1/2/3), this:
  1. Builds a target column shifted by that many hours
  2. Does a time-based train/test split
  3. Trains multiple candidate models, evaluates with RMSE/MAE/R2
  4. Saves the best model (as binary) + metadata to the MongoDB model registry,
     tagged with its horizon, so the dashboard can fetch "the Day 2 model"
     separately from "the Day 1 model".
  5. Cleans up old model versions, keeping only the most recent few per
     horizon, so the models collection doesn't grow unboundedly over time.

Run manually for now:
    python -m src.train

Later, GitHub Actions runs this daily.
"""

import io
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from bson.binary import Binary
from pymongo.errors import PyMongoError

from src import config, db

HORIZONS_HOURS = [24, 48, 72]
TRAIN_SPLIT_RATIO = 0.85

FEATURE_COLS = [
    "aqi", "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day", "month", "day_of_week", "aqi_change_rate",
]


def load_data(collection) -> pd.DataFrame:
    cursor = collection.find({"city": config.CITY_NAME}).sort("timestamp", 1)
    df = pd.DataFrame(list(cursor))
    df = df.drop(columns=[c for c in ["nh3", "_id"] if c in df.columns])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_target_and_split(df: pd.DataFrame, horizon_hours: int):
    """
    Build training examples where:
    - Pollutant features (aqi, pm2_5, pm10, co, no2, so2, o3, aqi_change_rate)
      come from time T (now) — the only pollution info actually available.
    - Weather features (temperature, humidity, pressure, wind_speed) come
      from time T+horizon — using the ACTUAL historical weather as a stand-in
      for what a forecast would say at real prediction time.
    - Calendar features (hour, day, month, day_of_week) also come from
      T+horizon, since these are always exactly knowable in advance.
    - Target is the AQI at T+horizon.

    Rows are aligned strictly by timestamp (not just row position), so any
    gaps in the hourly data (e.g. a missed feature-pipeline run) don't
    silently misalign "now" with the wrong "future" row.
    """
    n = len(df)
    if n <= horizon_hours:
        return pd.DataFrame(), pd.DataFrame()

    current = df.iloc[: n - horizon_hours].reset_index(drop=True)
    future = df.iloc[horizon_hours:].reset_index(drop=True)

    merged = pd.DataFrame({
        "current_timestamp": current["timestamp"].values,
        "future_timestamp": future["timestamp"].values,
        "aqi": current["aqi"].values,
        "pm2_5": current["pm2_5"].values,
        "pm10": current["pm10"].values,
        "co": current["co"].values,
        "no2": current["no2"].values,
        "so2": current["so2"].values,
        "o3": current["o3"].values,
        "aqi_change_rate": current["aqi_change_rate"].values,
        "temperature": future["temperature"].values,
        "humidity": future["humidity"].values,
        "pressure": future["pressure"].values,
        "wind_speed": future["wind_speed"].values,
        "hour": future["hour"].values,
        "day": future["day"].values,
        "month": future["month"].values,
        "day_of_week": future["day_of_week"].values,
        "aqi_target": future["aqi"].values,
    })

    # Strict alignment check: only keep rows where future_timestamp is
    # EXACTLY horizon_hours after current_timestamp. Protects against
    # silently mismatched rows if there were gaps in the hourly data.
    expected_future = pd.to_datetime(merged["current_timestamp"]) + pd.Timedelta(hours=horizon_hours)
    aligned_mask = pd.to_datetime(merged["future_timestamp"]) == expected_future
    merged = merged[aligned_mask].reset_index(drop=True)

    split_index = int(len(merged) * TRAIN_SPLIT_RATIO)
    train_df = merged.iloc[:split_index]
    test_df = merged.iloc[split_index:]
    return train_df, test_df


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train, y_train = train_df[FEATURE_COLS], train_df["aqi_target"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["aqi_target"]

    candidates = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
    }

    results = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
        }
        results.append({"name": name, "model": model, "metrics": metrics})

    return results


def select_best(results: list) -> dict:
    return max(results, key=lambda r: r["metrics"]["r2"])


def save_model_registry(best: dict, horizon_hours: int, models_collection):
    if best["name"] == "xgboost":
        # XGBoost recommends its own native format over generic pickling,
        # for compatibility across library versions.
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        best["model"].save_model(tmp_path)
        with open(tmp_path, "rb") as f:
            model_bytes = f.read()
        os.remove(tmp_path)
    else:
        buffer = io.BytesIO()
        joblib.dump(best["model"], buffer, compress=3)
        model_bytes = buffer.getvalue()

    size_mb = len(model_bytes) / (1024 * 1024)
    if size_mb > 15:  # MongoDB's hard limit is 16MB per document
        raise ValueError(
            f"Model for horizon {horizon_hours}h is {size_mb:.1f} MB even after compression — "
            f"too large to store in MongoDB. Consider reducing n_estimators/max_depth for this model."
        )
    trained_at = datetime.now(timezone.utc)

    metadata = {
        "city": config.CITY_NAME,
        "algorithm": best["name"],
        "trained_at": trained_at,
        "horizon_hours": horizon_hours,
        "feature_cols": FEATURE_COLS,
        "metrics": best["metrics"],
        "model_binary": Binary(model_bytes),
        "is_active": True,
    }

    models_collection.update_many(
        {"city": config.CITY_NAME, "horizon_hours": horizon_hours, "is_active": True},
        {"$set": {"is_active": False}},
    )
    models_collection.insert_one(metadata)

    size_kb = len(model_bytes) / 1024
    print(f"  Best: {best['name']} (R2={best['metrics']['r2']:.3f}), "
          f"serialized {size_kb:.1f} KB, saved as active {horizon_hours}h model")


def cleanup_old_models(models_collection, keep_n: int = 3):
    """
    Retention policy: for each (city, horizon), keep only the most recent
    `keep_n` model versions (by trained_at) and delete the rest. This
    prevents the models collection from growing unboundedly, since every
    training run adds a new binary (~500KB) per horizon and nothing
    would otherwise remove the old, inactive ones.
    """
    for horizon_hours in HORIZONS_HOURS:
        docs = list(
            models_collection.find(
                {"city": config.CITY_NAME, "horizon_hours": horizon_hours},
                {"_id": 1},
            ).sort("trained_at", -1)
        )
        ids_to_delete = [d["_id"] for d in docs[keep_n:]]
        if ids_to_delete:
            result = models_collection.delete_many({"_id": {"$in": ids_to_delete}})
            print(f"  Cleanup: removed {result.deleted_count} old model(s) for {horizon_hours}h horizon")


def run():
    print("Starting multi-horizon training pipeline...")
    config.validate()
    client = db.get_client()
    try:
        features_collection = db.get_collection(config.FEATURES_COLLECTION, client)
        models_collection = db.get_collection(config.MODELS_COLLECTION, client)

        df = load_data(features_collection)
        print(f"Loaded {len(df)} rows\n")

        for horizon_hours in HORIZONS_HOURS:
            print(f"--- Horizon: {horizon_hours}h (Day {horizon_hours // 24}) ---")
            train_df, test_df = build_target_and_split(df, horizon_hours)
            print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

            if len(train_df) < 50 or len(test_df) < 10:
                print(f"  Skipping {horizon_hours}h — not enough data yet "
                      f"(need at least ~{horizon_hours + 50} hours of history).\n")
                continue

            results = train_and_evaluate(train_df, test_df)
            for r in results:
                m = r["metrics"]
                print(f"  {r['name']}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, R2={m['r2']:.3f}")

            best = select_best(results)
            save_model_registry(best, horizon_hours, models_collection)
            print()

        print("Running cleanup...")
        cleanup_old_models(models_collection, keep_n=3)

        print("Multi-horizon training complete.")

    except PyMongoError as e:
        print(f"MongoDB error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    run()