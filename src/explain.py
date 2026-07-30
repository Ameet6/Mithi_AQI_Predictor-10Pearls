"""
SHAP explanation helpers.
Explains WHY a model produced a given prediction, by computing each
feature's contribution (positive = pushed AQI up, negative = pushed it down).

Uses shap.Explainer, which auto-selects the correct underlying algorithm
based on model type: exact tree-structure-based values for tree models
(Random Forest, Gradient Boosting, XGBoost), or coefficient-based values
for linear models (Linear Regression, Ridge) — both need a small
background sample of real data to establish a baseline ("what's a
typical/average prediction") to measure each feature's deviation from.
"""

import pandas as pd
import shap

FEATURE_LABELS = {
    "aqi": "Current AQI",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "co": "Carbon Monoxide",
    "no2": "Nitrogen Dioxide",
    "so2": "Sulfur Dioxide",
    "o3": "Ozone",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "wind_speed": "Wind Speed",
    "hour": "Hour of Day",
    "day": "Day of Month",
    "month": "Month",
    "day_of_week": "Day of Week",
    "aqi_change_rate": "AQI Trend",
}


def get_background_sample(features_collection, city: str, feature_cols: list, n: int = 200) -> pd.DataFrame:
    """
    Pull a small random sample of historical rows to use as SHAP's
    background reference — this establishes what a "typical" set of
    conditions looks like, so contributions can be measured as deviation
    from that baseline. Works the same way regardless of model type.
    """
    pipeline = [
        {"$match": {"city": city}},
        {"$sample": {"size": n}},
    ]
    docs = list(features_collection.aggregate(pipeline))
    df = pd.DataFrame(docs)
    return df[feature_cols]


def explain_prediction(model, feature_cols: list, feature_row: dict, background: pd.DataFrame) -> list:
    """
    Compute SHAP values for a single prediction, against a background
    sample for baseline comparison. Works for both tree-based and linear
    models via SHAP's auto-dispatching Explainer.

    Returns a list of dicts, sorted by absolute impact (largest first):
        [{"feature": "pressure", "value": 1004.2, "shap_value": 3.1}, ...]
    """
    X = pd.DataFrame([{col: feature_row.get(col) for col in feature_cols}])

    explainer = shap.Explainer(model, background)
    shap_values = explainer(X).values[0]

    contributions = [
        {"feature": feature_cols[i], "value": X.iloc[0, i], "shap_value": float(shap_values[i])}
        for i in range(len(feature_cols))
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return contributions