"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC dataset of Nairobi house listings.

NOTE: There is no single official public dataset of Nairobi house prices,
so this script builds a synthetic dataset using realistic parameters for
neighborhoods, sizes, and prices based on general Nairobi real-estate
market patterns (as of 2025/2026). Replace this with a real dataset
(e.g. scraped from Property24, BuyRentKenya, or your own data) for a
production-grade model — just keep the same column names.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Nairobi neighborhoods with an approximate relative price index
# (1.0 = baseline). Higher = more expensive area.
NEIGHBORHOODS = {
    "Karen":        2.6,
    "Runda":        2.8,
    "Muthaiga":     2.7,
    "Kitisuru":     2.5,
    "Lavington":    2.2,
    "Westlands":    2.0,
    "Kilimani":     1.9,
    "Kileleshwa":   1.8,
    "Loresho":      1.7,
    "Spring Valley":2.1,
    "Upperhill":    1.9,
    "South B":      1.1,
    "South C":      1.15,
    "Langata":      1.3,
    "Nyayo Estate": 1.1,
    "Embakasi":     0.75,
    "Donholm":      0.85,
    "Umoja":        0.7,
    "Kasarani":     0.7,
    "Roysambu":     0.8,
    "Ruaka":        1.0,
    "Kikuyu":       0.65,
    "Ngong Road":   1.3,
    "Rongai":       0.7,
    "Athi River":   0.55,
    "Thika Road":   0.8,
    "Ruiru":        0.6,
    "Buruburu":     0.9,
    "Kahawa":       0.6,
    "Dagoretti":    0.95,
}

PROPERTY_TYPES = ["Apartment", "Townhouse", "Bungalow", "Maisonette", "Villa"]

def generate(n=3000):
    rows = []
    neighborhoods = list(NEIGHBORHOODS.keys())

    for _ in range(n):
        neighborhood = np.random.choice(neighborhoods)
        price_index = NEIGHBORHOODS[neighborhood]

        property_type = np.random.choice(
            PROPERTY_TYPES, p=[0.55, 0.12, 0.10, 0.15, 0.08]
        )

        bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.10, 0.28, 0.32, 0.18, 0.08, 0.04])
        bathrooms = max(1, bedrooms - np.random.choice([0, 1], p=[0.6, 0.4]))

        # Size in square feet, correlated with bedrooms
        base_size = 350 + bedrooms * 280
        size_sqft = max(300, np.random.normal(base_size, base_size * 0.15))

        # Property-type size adjustment
        if property_type in ("Villa", "Bungalow"):
            size_sqft *= 1.25
        elif property_type == "Apartment":
            size_sqft *= 0.9

        age_years = np.random.randint(0, 30)  # building age
        distance_to_cbd_km = max(0.5, np.random.normal(15 / price_index, 5))

        has_parking = np.random.choice([0, 1], p=[0.2, 0.8])
        has_garden = 1 if property_type in ("Villa", "Bungalow", "Maisonette") and np.random.rand() > 0.4 else 0
        has_pool = 1 if price_index > 1.8 and np.random.rand() > 0.75 else 0
        gated_community = np.random.choice([0, 1], p=[0.35, 0.65])
        security_score = np.random.randint(1, 6)  # 1-5 rating

        # --- Price model (KES) ---
        price_per_sqft_base = 9000  # baseline KES per sqft
        price = size_sqft * price_per_sqft_base * price_index

        price *= (1 - age_years * 0.006)              # depreciation with age
        price *= (1 - distance_to_cbd_km * 0.006)      # distance discount
        price *= (1 + has_parking * 0.03)
        price *= (1 + has_garden * 0.04)
        price *= (1 + has_pool * 0.10)
        price *= (1 + gated_community * 0.06)
        price *= (1 + (security_score - 3) * 0.02)
        price *= (1 + bathrooms * 0.015)

        # random market noise
        price *= np.random.normal(1.0, 0.08)
        price = max(1_500_000, price)  # floor price

        rows.append({
            "neighborhood": neighborhood,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "size_sqft": round(size_sqft),
            "age_years": age_years,
            "distance_to_cbd_km": round(distance_to_cbd_km, 1),
            "has_parking": has_parking,
            "has_garden": has_garden,
            "has_pool": has_pool,
            "gated_community": gated_community,
            "security_score": security_score,
            "price_kes": round(price, -3),  # round to nearest 1000
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate(3000)
    df.to_csv("nairobi_housing.csv", index=False)
    print(f"Generated {len(df)} rows -> nairobi_housing.csv")
    print(df.head())
    print("\nPrice summary (KES):")
    print(df["price_kes"].describe())
