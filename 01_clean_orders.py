import pandas as pd

INPUT = "orders_full_merged.csv"
OUTPUT = "orders_clean.csv"

print("Lade Orders...")
df = pd.read_csv(INPUT)

# --- 1. Datumsfelder parsen ---
df["start"] = pd.to_datetime(df["gebuchte Einfahrtszeit"], errors="coerce")
df["end"] = pd.to_datetime(df["gebuchte Ausfahrtszeit"], errors="coerce")

# optional falls andere Spaltennamen existieren:
# df.rename(columns={"Einfahrtszeit": "start", "Ausfahrtszeit": "end"}, inplace=True)

# --- 2. Zeilen mit fehlenden Zeiten droppen ---
df = df.dropna(subset=["start", "end"])

# --- 3. end >= start ---
df = df[df["end"] > df["start"]]

# --- 4. Duplikate entfernen ---
df = df.drop_duplicates(subset=["Parkareal-ID", "start", "end"])

# --- 5. Areal-ID normalisieren ---
df["Parkareal-ID"] = df["Parkareal-ID"].astype(int)

# Ergebnis speichern
df.to_csv(OUTPUT, index=False)
print("FERTIG: orders_clean.csv gespeichert.")
print("Zeilen vorher:", len(df))
