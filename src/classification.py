
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


# 1. Last inn datasettene
ROOT = Path(__file__).resolve().parent.parent

weekdays = pd.read_csv(ROOT / "data" / "amsterdam_weekdays.csv")
weekends = pd.read_csv(ROOT / "data" / "amsterdam_weekends.csv")

df = pd.concat([weekdays, weekends], ignore_index=True)


# 2. Behold private rom og hele boliger
df = df[df["room_type"].isin([
    "Private room",
    "Entire home/apt"
])]


# 3. Velg variabler
X = df[["realSum", "bedrooms", "person_capacity", "dist"]]
y = df["room_type"]


# 4. Del i treningsdata og testdata
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 5. Standardiser dataene
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 6. Tren KNN-modellen
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# 7. Evaluer modellen
accuracy = accuracy_score(y_test, y_pred)

print("Klassifikasjonsresultater:")
print(f"Antall observasjoner: {len(df)}")
print(f"Accuracy: {accuracy * 100:.1f} %")


# 8. Lag confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["Private room", "Entire home/apt"],
    display_labels=["Privat rom", "Hel bolig"],
    cmap="Blues",
    colorbar=False
)

plt.title("KNN - Ukedager og helger")
plt.tight_layout()

figures = ROOT / "figures"
figures.mkdir(exist_ok=True)

plt.savefig(figures / "classification_combined.png", dpi=150)

plt.show()
