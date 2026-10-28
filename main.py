
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

bundle = joblib.load("models/model.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
FEATURE_COLUMNS = bundle["feature_columns"]
NUMERICAL_FEATURES = bundle["numerical_features"]
COUNTRIES = bundle["countries"]
FOOD_CATEGORIES = bundle["food_categories"]

TARGET = "Total Waste (Tons)"

# Map scaler statistics to their original column names.
scaler_means = dict(zip(scaler.feature_names_in_, scaler.mean_))
scaler_scales = dict(zip(scaler.feature_names_in_, scaler.scale_))

app = FastAPI(title="Global Food Waste Prediction API")


class FoodWasteRequest(BaseModel):
    country: str
    year: int = Field(ge=1900, le=2100)
    food_category: str
    economic_loss_million: float = Field(ge=0)
    avg_waste_per_capita: float = Field(ge=0)
    population_million: float = Field(ge=0)
    household_waste_percent: float = Field(ge=0, le=100)


@app.get("/")
def home():
    return {"message": "Global Food Waste Prediction API is ready"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: FoodWasteRequest):
    if data.country not in COUNTRIES:
        return {
            "error": "Unknown country",
            "allowed_values": COUNTRIES,
        }

    if data.food_category not in FOOD_CATEGORIES:
        return {
            "error": "Unknown food category",
            "allowed_values": FOOD_CATEGORIES,
        }

    # Build a row using the original dataset column names.
    row = pd.DataFrame([{
        "Country": data.country,
        "Year": data.year,
        "Food Category": data.food_category,
        "Economic Loss (Million $)": data.economic_loss_million,
        "Avg Waste per Capita (Kg)": data.avg_waste_per_capita,
        "Population (Million)": data.population_million,
        "Household Waste (%)": data.household_waste_percent,
    }])

    # Apply the same one-hot encoding as in the notebook.
    encoded = pd.get_dummies(
        row,
        columns=["Country", "Food Category"],
        drop_first=True,
    )

    # Match the exact feature names and order used during training.
    features = encoded.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0,
    ).copy()

    # Scale numerical input columns using the saved training scaler.
    for column in NUMERICAL_FEATURES:
        if column == TARGET:
            continue

        features[column] = (
            features[column] - scaler_means[column]
        ) / scaler_scales[column]

    # The notebook trained on a scaled target, so invert that scaling.
    scaled_prediction = float(model.predict(features)[0])

    prediction_tons = (
        scaled_prediction * scaler_scales[TARGET]
        + scaler_means[TARGET]
    )

    return {
        "predicted_total_waste_tons": round(prediction_tons, 2),
        "unit": "tons",
    }

