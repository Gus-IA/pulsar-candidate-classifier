"""Download the HTRU2 pulsar dataset from UCI and save it as a CSV."""

import pandas as pd
from ucimlrepo import fetch_ucirepo

htru2 = fetch_ucirepo(id=372)
df = pd.concat([htru2.data.features, htru2.data.targets], axis=1)
df = pd.read_csv("data/htru2.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df["class"].value_counts())
