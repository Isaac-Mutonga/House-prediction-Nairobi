"""
train_model.py
---------------
Trains and compares house price prediction models on the Nairobi
housing dataset, then saves the best model to disk for reuse.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["neighborhood", "property_type"]
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "size_sqft", "age_years",
    "distance_to_cbd_km", "has_parking", "has_garden",
    "has_pool", "gated_community", "security_score",
]
TARGET = "price_kes"


def load_data(path="nairobi_housing.csv"):
    return pd.read_csv(path)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    r2 = r2_score(y_test, preds)
    print(f"\n{name}")
    print(f"  MAE  : KES {mae:,.0f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R^2  : {r2:.4f}")
    return {"name": name, "mae": mae, "mape": mape, "r2": r2, "model": model}


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    results = []
    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        results.append(evaluate(name, pipe, X_test, y_test))

    best = min(results, key=lambda r: r["mape"])
    print(f"\nBest model: {best['name']} (lowest MAPE)")

    joblib.dump(best["model"], "best_model.pkl")
    print("Saved -> best_model.pkl")

    # Feature importance (only for tree-based models)
    if best["name"] in ("Random Forest", "Gradient Boosting"):
        model = best["model"].named_steps["model"]
        ohe = best["model"].named_steps["preprocessor"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = NUMERIC_FEATURES + cat_names
        importances = model.feature_importances_
        top = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
        print("\nTop 10 features driving price:")
        for feat, imp in top:
            print(f"  {feat:<25} {imp:.4f}")

    # Save a summary of all models for reference
    summary_df = pd.DataFrame(
        [{"model": r["name"], "MAE_KES": r["mae"], "MAPE_%": r["mape"], "R2": r["r2"]} for r in results]
    )
    summary_df.to_csv("model_comparison.csv", index=False)
    print("\nSaved comparison -> model_comparison.csv")


if __name__ == "__main__":
    main()
