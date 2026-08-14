

import joblib
import pandas as pd

MODEL_PATH = "best_model.pkl"


def predict_price(house: dict) -> float:
    """
    house must contain the following keys:
    neighborhood, property_type, bedrooms, bathrooms, size_sqft,
    age_years, distance_to_cbd_km, has_parking, has_garden,
    has_pool, gated_community, security_score
    """
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([house])
    price = model.predict(df)[0]
    return price


if __name__ == "__main__":
    new_house = {
        "neighborhood": "Kilimani",
        "property_type": "Apartment",
        "bedrooms": 3,
        "bathrooms": 2,
        "size_sqft": 1400,
        "age_years": 5,
        "distance_to_cbd_km": 4.5,
        "has_parking": 1,
        "has_garden": 0,
        "has_pool": 1,
        "gated_community": 1,
        "security_score": 4,
    }

    price = predict_price(new_house)
    print(f"Predicted price for this house: KES {price:,.0f}")
