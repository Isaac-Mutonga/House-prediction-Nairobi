import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from train_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, load_data

RANDOM_STATE = 42

df = load_data()
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

model = joblib.load("best_model.pkl")
preds = model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted
axes[0].scatter(y_test, preds, alpha=0.4, s=15, color="#2E86AB")
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
axes[0].plot(lims, lims, 'r--', linewidth=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual Price (KES)")
axes[0].set_ylabel("Predicted Price (KES)")
axes[0].set_title("Actual vs Predicted House Prices")
axes[0].legend()
axes[0].ticklabel_format(style='plain', axis='both')

# Average price by neighborhood (top 15)
avg_price = df.groupby("neighborhood")[TARGET].mean().sort_values(ascending=False).head(15)
axes[1].barh(avg_price.index[::-1], avg_price.values[::-1] / 1e6, color="#A23B72")
axes[1].set_xlabel("Average Price (KES, millions)")
axes[1].set_title("Top 15 Neighborhoods by Average Price")

plt.tight_layout()
plt.savefig("model_performance.png", dpi=150)
print("Saved -> model_performance.png")
