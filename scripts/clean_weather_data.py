import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------------
INPUT_XLSX  = "DWD_hourly_recent_3Months_filtered.xlsx"  # gefilterte Datei
OUTPUT_CSV  = "weather_hourly_clean.csv"

print(f"Lade {INPUT_XLSX} ...")
sheets = pd.read_excel(INPUT_XLSX, sheet_name=None)

if "Messungen" not in sheets:
    raise RuntimeError("Erwarte ein Sheet 'Messungen' in der gefilterten Datei.")

messungen = sheets["Messungen"].copy()

# ------------------------------------------------------------------
# 0. Basis-Aufräumen: stations_id & datetime
# ------------------------------------------------------------------
if "stations_id" not in messungen.columns:
    raise RuntimeError("Spalte 'stations_id' fehlt in Messungen.")

messungen["stations_id"] = messungen["stations_id"].astype(str).str.zfill(5)

if "datetime" not in messungen.columns:
    raise RuntimeError("Spalte 'datetime' fehlt in Messungen.")

messungen["datetime"] = pd.to_datetime(messungen["datetime"], errors="coerce")

# ------------------------------------------------------------------
# 1. Alle relevanten Klimavariablen definieren
# ------------------------------------------------------------------
numeric_cols = [
    "rr_mm",
    "temp_air_c",
    "dew_point_c",
    "rel_humidity_pct",
    "wind_speed_ms",
    "wind_dir_deg",
    "pressure_hpa",
    "sunshine_min",
    "solar_radiation_wm2",
    "cloud_cover_oktas",
]

# ggf. vorhandene auswählen
existing_numeric = [c for c in numeric_cols if c in messungen.columns]

# in numerische Typen casten (Komma → Punkt falls nötig)
for col in existing_numeric:
    messungen[col] = pd.to_numeric(
        messungen[col], errors="coerce"
    )

# ------------------------------------------------------------------
# 2. Sonne: NaN -> 0 (physikalisch Nachtstunden)
# ------------------------------------------------------------------
if "sunshine_min" in messungen.columns:
    messungen["sunshine_min"] = messungen["sunshine_min"].fillna(0)

# ------------------------------------------------------------------
# 3. Andere Variablen: stationweise Median-Imputation für Rest-NaNs
# ------------------------------------------------------------------
# (rr_mm würde ich meist so lassen, weil "kein Regen" ≠ 0, sondern „nicht gemessen“,
#  aber wenn du wirklich alle NaNs loswerden willst, kannst du es mit reinnehmen)

impute_cols = [
    "rel_humidity_pct",
    "cloud_cover_oktas",
    "temp_air_c",
    "dew_point_c",
    "wind_speed_ms",
    "wind_dir_deg",
    "pressure_hpa",
    "solar_radiation_wm2",
    # "rr_mm",  # optional, nur wenn du auch Regen-NaNs per Median füllen willst
]

for col in impute_cols:
    if col in messungen.columns:
        messungen[col] = messungen.groupby("stations_id")[col].transform(
            lambda x: x.fillna(x.median())
        )

# ------------------------------------------------------------------
# 4. Backfill-Flags entfernen ( *_orig_missing ), falls vorhanden
# ------------------------------------------------------------------
flag_cols = [c for c in messungen.columns if c.endswith("_orig_missing")]
if flag_cols:
    print(f"Entferne Backfill-Flag-Spalten: {flag_cols}")
    messungen = messungen.drop(columns=flag_cols)

# ------------------------------------------------------------------
# 5. Sortieren (stations_id, datetime)
# ------------------------------------------------------------------
messungen = messungen.sort_values(["stations_id", "datetime"])

# ------------------------------------------------------------------
# 6. Exportiere saubere Version
# ------------------------------------------------------------------
messungen.to_csv(OUTPUT_CSV, index=False)
print(f"Saubere Wetterdaten gespeichert unter: {OUTPUT_CSV}")
print(f"Zeilen: {len(messungen)}, Spalten: {len(messungen.columns)}")
