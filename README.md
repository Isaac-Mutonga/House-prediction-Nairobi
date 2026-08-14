# Nairobi House Price Prediction Model

A machine learning project that predicts residential house prices in Nairobi,
Kenya, based on property features and location.



## Project structure

```
nairobi_house_model/
├── nairobi_housing.csv    (3,000 listings)
├── train_model.py         # Trains & compares 3 models, saves the best
├── predict.py             # Predicts price for a new house
├── visualize.py           # Generates evaluation charts
├── best_model.pkl         # Saved trained model (Gradient Boosting)
├── model_comparison.csv   # Metrics for all models tried
├── model_performance.png  # Actual vs predicted + neighborhood price chart
└── README.md
```

## Features used

| Feature | Description |
|---|---|
| `neighborhood` | Area in Nairobi (30 neighborhoods included) |
| `property_type` | Apartment, Townhouse, Bungalow, Maisonette, Villa |
| `bedrooms` / `bathrooms` | Room counts |
| `size_sqft` | Floor area in square feet |
| `age_years` | Age of the building |
| `distance_to_cbd_km` | Distance to Nairobi CBD |
| `has_parking` / `has_garden` / `has_pool` | Amenities (0/1) |
| `gated_community` | Whether it's in a gated estate (0/1) |
| `security_score` | 1–5 rating |
| `price_kes` | Target variable — price in Kenyan Shillings |

## Models compared

| Model | MAE (KES) | MAPE | R² |
|---|---|---|---|
| Linear Regression | ~2.48M | 27.2% | 0.884 |
| Random Forest | ~2.22M | 18.3% | 0.897 |
| **Gradient Boosting (best)** | **~1.83M** | **15.4%** | **0.929** |

Gradient Boosting was automatically selected and saved as `best_model.pkl`.



## Predicting a new house price

Edit the `new_house` dictionary in `predict.py`, or use it as a function:

```python
from predict import predict_price

price = predict_price({
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
})
print(f"KES {price:,.0f}")
```

## Key insight from the model

Feature importance analysis shows that **size (sqft)** and **distance to the
CBD** are by far the strongest price drivers, followed by neighborhood
prestige (Runda, Karen, Muthaiga) and the presence of a pool.

