
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# 1. Last inn datasettene
ROOT = Path(__file__).resolve().parent.parent

weekdays = pd.read_csv(
    ROOT / "data" / "amsterdam_weekdays.csv"
)

weekends = pd.read_csv(
    ROOT / "data" / "amsterdam_weekends.csv"
)


# 2. Legg til hvilken periode dataene kommer fra
weekdays["period"] = "Weekday"
weekends["period"] = "Weekend"

# Kombiner datasettene
df = pd.concat([weekdays, weekends], ignore_index=True)


# 3. Sammenlign priser
print("Antall utleieobservasjoner:")
print(df["period"].value_counts())

print("\nPrisstatistikk:")
print(
    df.groupby("period")["realSum"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)


# 4. Beregn prisforskjellen
median_prices = df.groupby("period")["realSum"].median()

difference = (
    median_prices["Weekend"] - median_prices["Weekday"]
)

percentage = difference / median_prices["Weekday"] * 100

print(f"\nForskjell i medianpris: {difference:.2f} EUR")
print(f"Prosentvis forskjell: {percentage:.1f} %")


# 5. Sammenlign pris etter romtype
room_prices = df.groupby(
    ["room_type", "period"]
)["realSum"].median().unstack()

print("\nMedianpris etter romtype:")
print(room_prices.round(2))


# 6. Lag mappe for figurene
figures = ROOT / "figures"
figures.mkdir(exist_ok=True)


# Figur 1: Medianpris ukedag vs. helg
plt.figure(figsize=(8, 5))

median_prices.reindex(
    ["Weekday", "Weekend"]
).plot(kind="bar")

plt.ylabel("Medianpris for to netter (EUR)")
plt.xlabel("Periode")
plt.title("Airbnb-priser: Ukedager vs. helger")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(figures / "weekday_vs_weekend.png", dpi=150)


# Figur 2: Medianpris etter romtype og periode
room_prices.plot(kind="bar", figsize=(9, 5))

plt.ylabel("Medianpris for to netter (EUR)")
plt.xlabel("Romtype")
plt.title("Airbnb-priser etter romtype og periode")
plt.xticks(rotation=0)
plt.legend(title="Periode")
plt.tight_layout()

plt.savefig(figures / "price_roomtype_weekend.png", dpi=150)



# Antall utleier av hver romtype
print("\nAntall utleier etter romtype og periode:")
print(pd.crosstab(df["room_type"], df["period"]))

# Prosentvis fordeling av romtyper
print("\nProsentvis fordeling:")
print(
    pd.crosstab(
        df["room_type"],
        df["period"],
        normalize="columns"
    ).mul(100).round(1)
)


# Vis figurene
plt.show()
