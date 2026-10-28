
# test_api.py

from fastapi.testclient import TestClient
from main import app

# Reusable test client
client = TestClient(app)


def test_health_endpoint():
    """
    Verify that the API health-check endpoint is working.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_home_endpoint():
    """
    Verify that the API root endpoint is working.
    """
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_valid_food_waste_prediction():
    """
    Verify that the API accepts valid food-waste data
    and returns a numerical prediction.
    """

    payload = {
    "country": "India",
    "year": 2024,
    "food_category": "Fruits & Vegetables",
    "economic_loss_million": 100.0,
    "avg_waste_per_capita": 50.0,
    "population_million": 1400.0,
    "household_waste_percent": 30.0
}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_total_waste_tons" in result
    assert "unit" in result

    assert isinstance(result["predicted_total_waste_tons"], (int, float))
    assert result["unit"] == "tons"


def test_invalid_country():
    """
    Verify that an unsupported country is handled correctly.
    """
    payload = {
        "country": "InvalidCountry",
        "year": 2024,
        "food_category": "Vegetables",
        "economic_loss_million": 100.0,
        "avg_waste_per_capita": 50.0,
        "population_million": 1400.0,
        "household_waste_percent": 30.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "error" in response.json()


def test_invalid_household_waste_percentage():
    """
    Verify that invalid percentage values are rejected.
    """
    payload = {
        "country": "India",
        "year": 2024,
        "food_category": "Vegetables",
        "economic_loss_million": 100.0,
        "avg_waste_per_capita": 50.0,
        "population_million": 1400.0,
        "household_waste_percent": 150.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
