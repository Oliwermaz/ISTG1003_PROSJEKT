
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Last inn datasettene
ROOT = Path(__file__).resolve().parent.parent

weekdays = pd.read_csv(ROOT / "data" / "amsterdam_weekdays.csv")
weekends = pd.read_csv(ROOT / "data" / "amsterdam_weekends.csv")

# 0 = ukedag, 1 = helg
weekdays["weekend"] = 0
weekends["weekend"] = 1

# Kombiner dataene
df = pd.concat([weekdays, weekends], ignore_index=True)


# 2. Velg variabler
X = df[["dist", "bedrooms", "person_capacity", "weekend"]]
y = df["realSum"]


# 3. Del i treningsdata (80 %) og testdata (20 %)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Tren modellen
model = LinearRegression()
model.fit(X_train, y_train)

# Forutsi prisene
y_pred = model.predict(X_test)


# 5. Evaluer modellen
print("Regresjonsresultater:")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} EUR")

print("\nRegresjonskoeffisienter:")
print(f"Konstantledd: {model.intercept_:.2f}")

for name, coefficient in zip(X.columns, model.coef_):
    print(f"{name}: {coefficient:.2f}")


# 6. Opprett mappe for figurene
figures = ROOT / "figures"
figures.mkdir(exist_ok=True)


# 7. Faktisk pris mot forutsagt pris
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Faktisk pris (EUR)")
plt.ylabel("Forutsagt pris (EUR)")
plt.title("Lineær regresjon - Airbnb-priser")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures / "regression_combined.png", dpi=150)


# 8. Residualanalyse
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Forutsagt pris (EUR)")
plt.ylabel("Residual (EUR)")
plt.title("Residualanalyse - Kombinerte data")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures / "residuals_combined.png", dpi=150)

plt.show()
