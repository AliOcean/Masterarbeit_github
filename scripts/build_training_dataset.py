import pandas as pd

OCC_FILE   = "occupancy_hourly.csv"
MAP_FILE   = "parking_to_station_mapping.csv"
WEATHER_FILE = "weather_hourly_clean.csv"

OUT_FILE = "train_dataset_v1.csv"

# 1) Laden
occ = pd.read_csv(OCC_FILE)
mapping = pd.read_csv(MAP_FILE)
weather = pd.read_csv(WEATHER_FILE)

# 2) Spalten normalisieren
# occupancy_hourly.csv: timestamp, parkplatz_id, occupancy_category, occupancy_percent, hour, weekday
occ["timestamp"] = pd.to_datetime(occ["timestamp"], errors="coerce")
occ = occ.dropna(subset=["timestamp", "parkplatz_id"])

occ["parkplatz_id"] = occ["parkplatz_id"].astype(str)

# mapping: parking_id, stations_id
mapping["parking_id"] = mapping["parking_id"].astype(str)
mapping["stations_id"] = mapping["stations_id"].astype(str).str.zfill(5)

# weather: stations_id, datetime, rr_mm, temp_air_c ...
weather["stations_id"] = weather["stations_id"].astype(int).astype(str).str.zfill(5)
weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
weather = weather.dropna(subset=["stations_id", "datetime"])

# 3) Merge occupancy -> mapping
df = occ.merge(mapping[["parking_id", "stations_id"]],
               left_on="parkplatz_id",
               right_on="parking_id",
               how="left")

df = df.drop(columns=["parking_id"])

# 4) Merge -> weather (timestamp muss stündlich sein)
df = df.merge(
    weather[["stations_id", "datetime", "rr_mm", "temp_air_c"]],
    left_on=["stations_id", "timestamp"],
    right_on=["stations_id", "datetime"],
    how="left"
)

df = df.drop(columns=["datetime"])

# 5) Final columns + rename
df = df.rename(columns={
    "rr_mm": "rain",
    "temp_air_c": "temperature"
})

final = df[[
    "timestamp",
    "parkplatz_id",
    "occupancy_percent",
    "rain",
    "temperature",
    "hour",
    "weekday"
]].copy()

# Optional: unknown rauswerfen (occupancy_percent ist dann NaN)
final = final.dropna(subset=["occupancy_percent"])

final.to_csv(OUT_FILE, index=False)

print("✅ Fertig:", OUT_FILE)
print("Rows:", len(final))
print("Missing weather rain:", final["rain"].isna().mean())
print("Missing weather temp:", final["temperature"].isna().mean())
print(final.head())
