import pandas as pd
from pathlib import Path

# Finn datasettet
data_path = Path(__file__).resolve().parent.parent / "data" / "amsterdam_weekdays.csv"

# Les CSV-filen
df = pd.read_csv(data_path)

# Vis de første 5 radene
print(df.head())

# Vis antall rader og kolonner
print(df.shape)