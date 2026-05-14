import pandas as pd

INPUT = "ml_dataset_june_raw.csv"          # deine vollständige Datei
OUTPUT = "ml_dataset_june_labeled_full.csv"

print("Lade Datei:", INPUT)
df = pd.read_csv(INPUT, low_memory=False)

# --- 1) Zielvariable (Label) erzeugen: occupancy_class ---
def classify(r):
    try:
        r = float(r)
    except (TypeError, ValueError):
        return None

    if r < 0.40:
        return "low"
    elif r < 0.75:
        return "medium"
    else:
        return "high"

if "occupancy_ratio" not in df.columns:
    raise ValueError("Spalte 'occupancy_ratio' wurde in der Input-Datei nicht gefunden!")

df["occupancy_class"] = df["occupancy_ratio"].apply(classify)

print("Anzahl Zeilen:", len(df))
print("Einzigartige Labels:", df["occupancy_class"].unique())

# --- 2) NICHTS löschen, alles behalten ---
df.to_csv(OUTPUT, index=False)
print("\nFERTIG.")
print(f"Neue Datei gespeichert als: {OUTPUT}")
print("Spalten gesamt:", len(df.columns))
