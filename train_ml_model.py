import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1) Dataset laden
# =========================
INPUT_FILE = "ml_dataset_june_labeled_full.csv"  # ggf. anpassen
print(f"Lade {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

# Sicherstellen, dass occupancy_class existiert
if "occupancy_class" not in df.columns:
    raise ValueError("Spalte 'occupancy_class' fehlt im Dataset!")

# =========================
# 2) Zielvariable y definieren
# =========================
y = df["occupancy_class"]

# =========================
# 3) Feature-Matrix X vorbereiten
# =========================
X = df.drop(columns=["occupancy_class"])

# timestamp in datetime wandeln (für Sortierung / ggf. später Features)
if "timestamp" in X.columns:
    X["timestamp"] = pd.to_datetime(X["timestamp"], errors="coerce")

# Boolean-/Yes/No-Spalten in 0/1 umwandeln
for col in X.columns:
    if X[col].dtype == object:
        uniq = set(X[col].dropna().unique())
        # klassische Yes/No-Felder
        if uniq.issubset({"Yes", "No", "yes", "no", "Y", "N"}):
            X[col] = X[col].str.lower().map(
                {"yes": 1, "no": 0, "y": 1, "n": 0}
            )
        # True/False als Text
        elif uniq.issubset({"True", "False", True, False}):
            X[col] = X[col].astype(str).map(
                {"True": 1, "False": 0}
            )

# Nach Konvertierung: nur numerische Features behalten
X_num = X.select_dtypes(include=[np.number]).copy()

print("Anzahl Features (numerisch):", X_num.shape[1])
print("Anzahl Zeilen:", X_num.shape[0])

# =========================
# 4) Train/Test-Split (zeitbasiert grob)
# =========================
# Wenn timestamp existiert, danach sortieren, damit wir nicht komplett mischen
if "timestamp" in df.columns:
    df_sorted = df.sort_values("timestamp")
    y = df_sorted["occupancy_class"]
    X_num = X_num.loc[df_sorted.index]

# 70% Training, 30% Test
X_train, X_test, y_train, y_test = train_test_split(
    X_num, y, test_size=0.3, shuffle=False  # kein Shuffle -> zeitlich „realistischer“
)

print("Train-Shape:", X_train.shape)
print("Test-Shape :", X_test.shape)

# =========================
# 5) Modell trainieren (Random Forest als Baseline)
# =========================
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("Trainiere RandomForest...")
clf.fit(X_train, y_train)

# =========================
# 6) Auswertung
# =========================
y_pred = clf.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# einfache Feature-Importance-Ausgabe
importances = clf.feature_importances_
feat_importance = sorted(
    zip(X_num.columns, importances), key=lambda x: x[1], reverse=True
)

print("\nTop 20 wichtigste Features:")
for name, score in feat_importance[:20]:
    print(f"{name:35s} {score:.4f}")
