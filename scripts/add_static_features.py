import pandas as pd

TRAIN_FILE = "train_dataset_v1.csv"
STATIC_FILE = "parking_coordinates.csv"

OUT_FILE = "train_dataset_final.csv"

train = pd.read_csv(TRAIN_FILE)
static = pd.read_csv(STATIC_FILE)

static["parking_id"] = static["parking_id"].astype(str)
train["parkplatz_id"] = train["parkplatz_id"].astype(str)

df = train.merge(
    static[["parking_id", "capacity", "latitude", "longitude"]],
    left_on="parkplatz_id",
    right_on="parking_id",
    how="left"
)

df = df.drop(columns=["parking_id"])

df.to_csv(OUT_FILE, index=False)

print("Fertig:", OUT_FILE)
print("Rows:", len(df))
print(df.head())
