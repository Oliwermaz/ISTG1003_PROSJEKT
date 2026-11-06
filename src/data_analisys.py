
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. LAST INN DATASETTET
# ============================================================

# Finn prosjektmappen
ROOT = Path(__file__).resolve().parent.parent

# Les Airbnb-datasettet
df = pd.read_csv(ROOT / "data" / "amsterdam_weekdays.csv")


# ============================================================
# 2. UTFORSKENDE DATAANALYSE (EDA)
# ============================================================

# Undersøk størrelsen på datasettet
print("\nAntall rader og kolonner:")
print(df.shape)

# Vis alle kolonnene
print("\nKolonner:")
print(df.columns.tolist())

# Vis de første fem radene
print("\nDe første fem radene:")
print(df.head())

# Undersøk manglende verdier
print("\nManglende verdier:")
print(df.isnull().sum())

# Statistisk sammendrag
print("\nStatistisk sammendrag:")
print(
    df[["realSum", "dist", "bedrooms", "person_capacity"]]
    .describe()
)

# Undersøk romtypene
print("\nAntall utleier per romtype:")
print(df["room_type"].value_counts())


# ============================================================
# 3. OPPRETT MAPPE FOR FIGURER
# ============================================================

figures = ROOT / "figures"
figures.mkdir(exist_ok=True)


# ============================================================
# 4. FIGUR 1 - PRIS MOT AVSTAND TIL SENTRUM
# ============================================================

plt.figure(figsize=(9, 5))

plt.scatter(
    df["dist"],
    df["realSum"],
    alpha=0.3
)

plt.xlabel("Avstand til sentrum (km)")
plt.ylabel("Pris for to netter (EUR)")
plt.title("Airbnb-pris mot avstand til sentrum")

plt.grid(alpha=0.3)
plt.tight_layout()

# Lagre figuren
plt.savefig(
    figures / "price_vs_distance.png",
    dpi=150
)


# ============================================================
# 5. FIGUR 2 - PRISFORDELING
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(
    df["realSum"],
    bins=50,
    edgecolor="black"
)

plt.xlabel("Pris for to netter (EUR)")
plt.ylabel("Antall utleier")
plt.title("Fordeling av Airbnb-priser")

plt.grid(alpha=0.3)
plt.tight_layout()

# Lagre figuren
plt.savefig(
    figures / "price_distribution.png",
    dpi=150
)


# ============================================================
# 6. FIGUR 3 - PRIS ETTER ROMTYPE
# ============================================================

plt.figure(figsize=(9, 5))

# Hent romtypene
room_types = df["room_type"].unique()

# Hent prisene for hver romtype
room_data = [
    df.loc[df["room_type"] == room, "realSum"]
    for room in room_types
]

# Opprett boksplott
plt.boxplot(
    room_data,
    tick_labels=room_types,
    showfliers=False
)

plt.xlabel("Romtype")
plt.ylabel("Pris for to netter (EUR)")
plt.title("Prisfordeling etter romtype")

plt.grid(alpha=0.3)
plt.tight_layout()

# Lagre figuren
plt.savefig(
    figures / "price_by_roomtype.png",
    dpi=150
)


# ============================================================
# 7. KORRELASJONSANALYSE
# ============================================================

print("\nKorrelasjon med pris:")

columns = [
    "realSum",
    "dist",
    "bedrooms",
    "person_capacity"
]

# Beregn Pearson-korrelasjon
correlation = df[columns].corr()

# Vis korrelasjonen mellom pris og øvrige variabler
print(correlation["realSum"].round(3))


# ============================================================
# 8. UNDERSØK EKSTREME PRISER
# ============================================================

print("\nAntall utleier over 2000 EUR:")
print((df["realSum"] > 2000).sum())

print("\nDe fem dyreste utleiene:")
print(
    df[["realSum", "dist", "bedrooms", "person_capacity"]]
    .sort_values(by="realSum", ascending=False)
    .head()
)


# ============================================================
# 9. VIS ALLE FIGURENE
# ============================================================

print("\nAnalysen er fullført!")
print("Figurene er lagret i figures-mappen.")

# Vis alle figurene samtidig
plt.show()
