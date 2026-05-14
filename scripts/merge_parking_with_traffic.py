# merge_parking_with_traffic.py
import pandas as pd

parking_file = "train_dataset_final.csv"
traffic_file = "bast_traffic_hourly_2025-11_2025-12.csv"
mapping_file = "parkplatz_to_bast_station_mapping_enriched.csv"

output_file = "train_dataset_final_with_traffic_full.csv"


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    dupes = df.columns[df.columns.duplicated()].unique().tolist()
    if dupes:
        print("WARNING: Duplicate columns found and removed:", dupes)
    return df.loc[:, ~df.columns.duplicated()].copy()


print("Loading parking dataset...")
parking = pd.read_csv(parking_file, parse_dates=["timestamp"])
parking = drop_duplicate_columns(parking)

print("Loading traffic dataset...")
traffic = pd.read_csv(traffic_file, parse_dates=["timestamp"])
traffic = drop_duplicate_columns(traffic)

print("Loading mapping dataset...")
mapping = pd.read_csv(mapping_file)
mapping = drop_duplicate_columns(mapping)

# Mapping-Spalte normieren
if "nearest_station_id" in mapping.columns and "station_id" not in mapping.columns:
    mapping = mapping.rename(columns={"nearest_station_id": "station_id"})

keep_cols = [c for c in ["parkplatz_id", "station_id", "nearest_station_dist_km", "road_class", "road_number"] if c in mapping.columns]
mapping = mapping[keep_cols].drop_duplicates(subset=["parkplatz_id"])

# safety: falls parking schon station_id hat -> entfernen
if "station_id" in parking.columns:
    print("INFO: 'station_id' exists in parking dataset -> dropping it before merge.")
    parking = parking.drop(columns=["station_id"])

print("Merging parking with station mapping (LEFT)...")
df = parking.merge(mapping, on="parkplatz_id", how="left")
df = drop_duplicate_columns(df)

print("Merging traffic data (LEFT on station_id+timestamp)...")
df = df.merge(traffic, on=["station_id", "timestamp"], how="left")
df = drop_duplicate_columns(df)

print("Calculating sv_share...")
df["sv_share"] = df["sv_total"] / df["kfz_total"]

print("Saving...")
df.to_csv(output_file, index=False)

print("Done.")
print("Rows:", len(df))
print("Traffic coverage (kfz_total notna):", df["kfz_total"].notna().mean())
