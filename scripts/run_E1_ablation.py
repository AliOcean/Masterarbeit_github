# run_E1_ablation.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# CONFIG
# =========================
INPUT_FILE = "train_dataset_final_with_traffic_full.csv"

OUT_DIR = Path("E1_results")
OUT_DIR.mkdir(exist_ok=True)

# Zielspalte (anpassen, falls bei dir anders)
TARGET_COL = "occupancy_percent"

# Zeitfenster E1: Nov/Dez
E1_START = "2025-11-01"
E1_END = "2026-01-01"

# Zeitbasierter Split (kein Leakage)
# Train <= SPLIT_DATE, Test > SPLIT_DATE
SPLIT_DATE = "2025-12-15 23:00:00"

# Random Forest Settings (solide Defaults)
RF_PARAMS = dict(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)

# Traffic Features
TRAFFIC_COLS = ["kfz_total", "sv_total", "sv_share"]

# Spalten, die wir nie als Features nutzen wollen
ALWAYS_DROP = [
    TARGET_COL,
]

# =========================
# HELPERS
# =========================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "N": int(len(y_true)),
    }


def select_feature_columns(df: pd.DataFrame, include_traffic: bool) -> list[str]:
    # Grundidee: wir nehmen alle numerischen Features,
    # außer Ziel, und optional ohne Traffic-Spalten.

    # Nur numerische Spalten
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Ziel raus
    features = [c for c in num_cols if c not in ALWAYS_DROP]

    if not include_traffic:
        features = [c for c in features if c not in TRAFFIC_COLS]

    # Falls station_id oder road_number als int drin ist, ist das okay.
    # Aber: wenn du das NICHT willst, kannst du sie hier entfernen.
    # Beispiel:
    # features = [c for c in features if c not in ["station_id", "road_number"]]

    return features


def train_and_eval(df_train, df_test, features, label, model_name: str):
    X_train = df_train[features]
    y_train = df_train[label].astype(float)

    X_test = df_test[features]
    y_test = df_test[label].astype(float)

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    m_train = metrics_dict(y_train, pred_train)
    m_test = metrics_dict(y_test, pred_test)

    # Feature Importances
    fi = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # Predictions speichern
    pred_df = df_test[["timestamp", "parkplatz_id"]].copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = pred_test
    pred_df.to_csv(OUT_DIR / f"predictions_{model_name}.csv", index=False)

    fi.to_csv(OUT_DIR / f"feature_importance_{model_name}.csv", index=False)

    return model, m_train, m_test, fi


# =========================
# MAIN
# =========================
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

# Basic sanity
required = {"timestamp", "parkplatz_id", TARGET_COL}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Filtering E1 window (Nov/Dec) ...")
df = df[(df["timestamp"] >= E1_START) & (df["timestamp"] < E1_END)].copy()

print("Keeping only rows with Traffic (E1 subset) ...")
# Wichtig: E1-Subset = nur Zeilen, wo Traffic wirklich existiert
df = df[df["kfz_total"].notna()].copy()

print("E1 rows:", len(df))
print("Unique parkplaetze:", df["parkplatz_id"].nunique())
print("Time range:", df["timestamp"].min(), "->", df["timestamp"].max())

# Zeitbasierter Split
split_ts = pd.to_datetime(SPLIT_DATE)
train = df[df["timestamp"] <= split_ts].copy()
test = df[df["timestamp"] > split_ts].copy()

print("Train rows:", len(train), "Test rows:", len(test))
if len(train) == 0 or len(test) == 0:
    raise ValueError("Train/Test split resulted in empty set. Adjust SPLIT_DATE.")

# Featuresets
features_no_traffic = select_feature_columns(df, include_traffic=False)
features_with_traffic = select_feature_columns(df, include_traffic=True)

# Sicherstellen, dass Traffic-Spalten wirklich drin sind im 'with_traffic'
for c in TRAFFIC_COLS:
    if c not in features_with_traffic:
        print(f"WARNING: Traffic col {c} not in numeric features (maybe non-numeric?).")

print("Num features (no traffic):", len(features_no_traffic))
print("Num features (with traffic):", len(features_with_traffic))

# Train/Eval ohne Traffic
print("\nTraining model WITHOUT traffic...")
_, m_train_no, m_test_no, fi_no = train_and_eval(
    train, test, features_no_traffic, TARGET_COL, model_name="rf_no_traffic"
)

# Train/Eval mit Traffic
print("\nTraining model WITH traffic...")
_, m_train_yes, m_test_yes, fi_yes = train_and_eval(
    train, test, features_with_traffic, TARGET_COL, model_name="rf_with_traffic"
)

# Ergebnisse speichern
summary = {
    "config": {
        "INPUT_FILE": INPUT_FILE,
        "TARGET_COL": TARGET_COL,
        "E1_START": E1_START,
        "E1_END": E1_END,
        "SPLIT_DATE": SPLIT_DATE,
        "RF_PARAMS": RF_PARAMS,
        "TRAFFIC_COLS": TRAFFIC_COLS,
    },
    "subset_stats": {
        "rows": int(len(df)),
        "unique_parkplaetze": int(df["parkplatz_id"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    },
    "metrics": {
        "rf_no_traffic": {
            "train": m_train_no,
            "test": m_test_no,
        },
        "rf_with_traffic": {
            "train": m_train_yes,
            "test": m_test_yes,
        },
    }
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

summary_table = pd.DataFrame([
    {"model": "rf_no_traffic", **m_test_no},
    {"model": "rf_with_traffic", **m_test_yes},
])
summary_table.to_csv(OUT_DIR / "summary_table.csv", index=False)

print("\n=== TEST METRICS (E1) ===")
print(summary_table.to_string(index=False))

print("\nSaved outputs in:", OUT_DIR.resolve())
print("Files:")
print("- summary.json")
print("- summary_table.csv")
print("- predictions_rf_no_traffic.csv")
print("- predictions_rf_with_traffic.csv")
print("- feature_importance_rf_no_traffic.csv")
print("- feature_importance_rf_with_traffic.csv")
