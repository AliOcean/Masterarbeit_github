import re
import pandas as pd

# ------------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------------
# Passe diese Namen an dein Export-Skript an
INPUT_XLSX  = "DWD_hourly_recent_3Months.xlsx"
OUTPUT_XLSX = "DWD_hourly_recent_3Months_filtered.xlsx"

# Coverage-Kriterien (kannst du jederzeit anpassen)
THRESHOLDS = {
    "cov_temp_air_c":      0.6,
    "cov_wind_speed_ms":   0.6,
    "cov_pressure_hpa":    0.6,
    # "cov_sunshine_min":    0.6,
    # falls du z.B. Bewölkung mit reinnehmen willst:
    # "cov_cloud_cover_oktas": 0.7,
}

# ------------------------------------------------------------------
# Laden der Excel-Datei (alle Sheets)
# ------------------------------------------------------------------
print(f"Lade {INPUT_XLSX} ...")
sheets = pd.read_excel(INPUT_XLSX, sheet_name=None)

if "Stationen" not in sheets:
    raise RuntimeError("Erwarte ein Sheet 'Stationen' in der Eingabedatei.")

if "Coverage" not in sheets:
    raise RuntimeError("Es gibt kein Sheet 'Coverage'. Bitte das 2.2/2.3-Hauptskript benutzen, das Coverage schreibt.")

stationen = sheets["Stationen"]
coverage  = sheets["Coverage"]

# ---- alle Messungen-Sheets einsammeln (Messungen, Messungen_2, Messungen_3, ...) ----
mess_sheet_names = [name for name in sheets.keys() if name.startswith("Messungen")]

if not mess_sheet_names:
    raise RuntimeError("Es wurden keine Sheets gefunden, die mit 'Messungen' beginnen.")

def sort_key(name: str) -> int:
    """
    Sorgt dafür, dass die Reihenfolge sinnvoll ist:
      Messungen, Messungen_2, Messungen_3, ...
    """
    if name == "Messungen":
        return 0
    m = re.match(r"Messungen_(\d+)$", name)
    if m:
        return int(m.group(1))
    return 999999  # falls irgendwas Sonderbares auftaucht

mess_sheet_names_sorted = sorted(mess_sheet_names, key=sort_key)

print("Gefundene Messungen-Sheets:")
for n in mess_sheet_names_sorted:
    print(f"  - {n}")

# Alle Messungen-Sheets vertikal zusammenfügen
messungen_list = [sheets[name] for name in mess_sheet_names_sorted]
messungen = pd.concat(messungen_list, ignore_index=True)
print(f"Gesamtzeilen Messungen (vor Filter): {len(messungen)}")

# ------------------------------------------------------------------
# Coverage-Filter auf Stationen anwenden
# ------------------------------------------------------------------
print("Wende Coverage-Filter an ...")

cov = coverage.copy()
cov["stations_id"] = cov["stations_id"].astype(str).str.zfill(5)

mask = pd.Series(True, index=cov.index)
for col, thr in THRESHOLDS.items():
    if col not in cov.columns:
        print(f"[Hinweis] Spalte {col} fehlt in Coverage – wird ignoriert.")
        continue
    mask &= cov[col] >= thr

selected_cov = cov[mask].copy()
selected_ids  = selected_cov["stations_id"].unique()

print(f"Ausgewählte Stationen: {len(selected_ids)}")

if len(selected_ids) == 0:
    raise RuntimeError("Keine Station erfüllt die Coverage-Kriterien – Schwellen evtl. zu streng?")

# ------------------------------------------------------------------
# Messungen und Stationen filtern
# ------------------------------------------------------------------
print("Filtere Messungen und Stationen ...")

messungen_f = messungen.copy()
messungen_f["stations_id"] = messungen_f["stations_id"].astype(str).str.zfill(5)
messungen_f = messungen_f[messungen_f["stations_id"].isin(selected_ids)].copy()

stationen_f = stationen.copy()
stationen_f["stations_id"] = stationen_f["stations_id"].astype(str).str.zfill(5)
stationen_f = stationen_f[stationen_f["stations_id"].isin(selected_ids)].copy()

coverage_f = coverage.copy()
coverage_f["stations_id"] = coverage_f["stations_id"].astype(str).str.zfill(5)
coverage_f = coverage_f[coverage_f["stations_id"].isin(selected_ids)].copy()

print(f"Gefilterte Messungs-Zeilen: {len(messungen_f)}")
print(f"Gefilterte Stationen:        {len(stationen_f)}")

# ------------------------------------------------------------------
# Neue Excel-Datei schreiben
# ------------------------------------------------------------------
print(f"Schreibe {OUTPUT_XLSX} ...")

with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as writer:
    stationen_f.to_excel(writer, sheet_name="Stationen", index=False)
    # Wichtig: jetzt wieder EIN Messungen-Sheet
    messungen_f.to_excel(writer, sheet_name="Messungen", index=False)
    coverage_f.to_excel(writer, sheet_name="Coverage", index=False)

print("Fertig.")
