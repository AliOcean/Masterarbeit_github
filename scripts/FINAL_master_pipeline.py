# FINAL_master_pipeline.py
# One script to rule them all: RF_full, RF_no_lag1h, XGB_full + plots + SHAP + comparison table
# Python 3.11

from __future__ import annotations

import os
import json
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

import joblib

# Optional libs
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# =========================
# CONFIG (EDIT HERE ONLY)
# =========================
DATA_PATH = r"G:\Meine Ablage\Universität\Masterarbeit\Code\train_dataset_final.csv"  # can be .csv or .csv.gz
OUT_DIR = r"G:\Meine Ablage\Universität\Masterarbeit\Code\FINAL_MASTER_RESULTS"

TARGET_COL = "occupancy_percent"
TIME_COL = "timestamp"
ID_COL = "parkplatz_id"

# Base columns you already have in train_dataset_final.csv
BASE_FEATURES = [
    "rain", "temperature", "hour", "weekday", "capacity", "latitude", "longitude"
]
# We will add: month, is_weekend, lag_1h, lag_24h, lag_168h, rolling_mean_3h, rolling_std_3h, rolling_mean_24h, rolling_std_24h

# Train/test split
TEST_FRACTION = 0.20  # time-based split (last 20% timestamps as test)

# Feature engineering settings
LAGS_HOURS = [1, 24, 168]
ROLL_WINDOWS = [3, 24]

# Sampling for heavy plots (scatter etc.)
SCATTER_SAMPLE_N = 50_000  # reduce if slow

# SHAP
DO_SHAP = True
SHAP_MODEL_NAME = "RF_full"          # only run SHAP for this model to keep runtime sane
SHAP_SAMPLES = 200                   # your choice
SHAP_APPROXIMATE = True              # makes it way faster (recommended)
SHAP_CHECK_ADDITIVITY = False        # faster + avoids warnings

# Models
RF_PARAMS = dict(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

# XGBoost params (reasonable defaults)
XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42
)


# =========================
# HELPERS
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Avoid division by zero in MAPE
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

    medae = float(np.median(np.abs(y_true - y_pred)))

    # Extra nice metric for thesis: P90 absolute error
    p90ae = float(np.quantile(np.abs(y_true - y_pred), 0.90))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "MedianAE": medae,
        "P90AE": p90ae,
        "N": int(len(y_true))
    }

def plot_and_save(figpath: Path) -> None:
    plt.tight_layout()
    plt.savefig(figpath, dpi=200)
    plt.close()

def sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)

def read_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {path}")

    if p.suffix.lower() == ".gz":
        df = pd.read_csv(p, compression="gzip")
    else:
        df = pd.read_csv(p)

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["month"] = df[TIME_COL].dt.month.astype(int)
    df["is_weekend"] = (df["weekday"] >= 6).astype(int)  # assuming 1..7 or 0..6? We'll handle below.
    return df

def normalize_weekday(df: pd.DataFrame) -> pd.DataFrame:
    # Your dataset: weekday looked like 7 for Sunday. Make consistent:
    # If weekday in [1..7], weekend = 6/7; If in [0..6], weekend = 5/6
    df = df.copy()
    w = df["weekday"]
    if w.min() >= 1 and w.max() <= 7:
        df["is_weekend"] = df["weekday"].isin([6, 7]).astype(int)
    else:
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df

def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([ID_COL, TIME_COL])

    # Lags
    for h in LAGS_HOURS:
        df[f"lag_{h}h"] = df.groupby(ID_COL)[TARGET_COL].shift(h)

    # Rolling stats (based on past values, shift by 1 to avoid leakage)
    for w in ROLL_WINDOWS:
        grp = df.groupby(ID_COL)[TARGET_COL]
        rolled = grp.shift(1).rolling(window=w, min_periods=1)
        df[f"rolling_mean_{w}h"] = rolled.mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{w}h"] = rolled.std().reset_index(level=0, drop=True)

    return df

def time_based_split(df: pd.DataFrame, test_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(TIME_COL)
    unique_times = df[TIME_COL].dropna().sort_values().unique()
    cut_idx = int((1.0 - test_fraction) * len(unique_times))
    cut_time = unique_times[max(1, min(cut_idx, len(unique_times)-1))]
    train = df[df[TIME_COL] < cut_time].copy()
    test = df[df[TIME_COL] >= cut_time].copy()
    return train, test

def get_feature_list(include_lag1h: bool = True) -> List[str]:
    feats = []
    feats.extend(BASE_FEATURES)
    feats.extend(["month", "is_weekend"])
    # lags
    for h in LAGS_HOURS:
        if (h == 1) and (not include_lag1h):
            continue
        feats.append(f"lag_{h}h")
    # rolling
    for w in ROLL_WINDOWS:
        feats.append(f"rolling_mean_{w}h")
        feats.append(f"rolling_std_{w}h")
    return feats

def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train: np.ndarray, y_train: np.ndarray):
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model

def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    return pd.DataFrame({"feature": feature_names, "importance": np.nan})

def make_common_plots(
    out_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_df_for_time: Optional[pd.DataFrame] = None
) -> None:
    # prediction vs actual (scatter sample)
    dfp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    dfp_s = sample_df(dfp, SCATTER_SAMPLE_N)

    plt.figure()
    plt.scatter(dfp_s["y_true"], dfp_s["y_pred"], s=3)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Prediction vs Actual - {model_name}")
    plot_and_save(out_dir / f"{model_name}_prediction_vs_actual.png")

    # residual histogram
    resid = y_pred - y_true
    plt.figure()
    plt.hist(resid, bins=80)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution - {model_name}")
    plot_and_save(out_dir / f"{model_name}_residual_histogram.png")

    # residual vs predicted
    df_r = pd.DataFrame({"pred": y_pred, "resid": resid})
    df_r_s = sample_df(df_r, SCATTER_SAMPLE_N)
    plt.figure()
    plt.scatter(df_r_s["pred"], df_r_s["resid"], s=3)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residual vs Predicted - {model_name}")
    plot_and_save(out_dir / f"{model_name}_residual_vs_predicted.png")

    # cumulative absolute error curve (CDF)
    abs_err = np.abs(resid)
    abs_err_sorted = np.sort(abs_err)
    cdf = np.arange(1, len(abs_err_sorted) + 1) / len(abs_err_sorted)
    plt.figure()
    plt.plot(abs_err_sorted, cdf)
    plt.xlabel("Absolute Error")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Cumulative Error Distribution - {model_name}")
    plot_and_save(out_dir / f"{model_name}_cumulative_error.png")

    # residual vs time (only if timestamps available)
    if pred_df_for_time is not None and TIME_COL in pred_df_for_time.columns:
        # sample to avoid spaghetti
        dft = pred_df_for_time[[TIME_COL, "resid"]].dropna()
        dft = dft.sort_values(TIME_COL)
        dft_s = sample_df(dft, min(80_000, len(dft)))
        plt.figure()
        plt.plot(dft_s[TIME_COL], dft_s["resid"])
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.title(f"Residual vs Time - {model_name} (sample)")
        plot_and_save(out_dir / f"{model_name}_residual_vs_time.png")


def run_shap(
    out_dir: Path,
    model_name: str,
    model,
    X_sample: np.ndarray,
    feature_names: List[str]
) -> None:
    if not SHAP_AVAILABLE:
        print("SHAP not available -> skipping.")
        return

    # TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)

    t0 = time.time()
    shap_values = explainer.shap_values(
        X_sample,
        approximate=SHAP_APPROXIMATE,
        check_additivity=SHAP_CHECK_ADDITIVITY
    )
    dt = time.time() - t0
    print(f"SHAP computed for {model_name} in {dt:.1f}s on {len(X_sample)} samples.")

    # bar (mean |shap|)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plot_and_save(out_dir / f"{model_name}_shap_bar.png")

    # beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plot_and_save(out_dir / f"{model_name}_shap_summary.png")

    # local waterfall for first sample
    try:
        i = 0
        # For regression, shap_values is (n_samples, n_features)
        sv = shap_values[i]
        base = explainer.expected_value
        exp = shap.Explanation(values=sv, base_values=base, data=X_sample[i], feature_names=feature_names)
        plt.figure()
        shap.plots.waterfall(exp, show=False)
        plot_and_save(out_dir / f"{model_name}_shap_local_waterfall.png")
    except Exception as e:
        print("Waterfall plot failed:", e)


# =========================
# MAIN PIPELINE
# =========================
def main() -> None:
    out_root = Path(OUT_DIR)
    ensure_dir(out_root)

    print("Loading dataset...")
    df = read_dataset(DATA_PATH)

    # Basic checks
    needed_cols = [TIME_COL, ID_COL, TARGET_COL] + BASE_FEATURES
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Parse + feature engineering
    print("Adding time features...")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = add_time_features(df)
    df = normalize_weekday(df)

    print("Adding lag/rolling features (can take time)...")
    df = add_lag_rolling_features(df)

    # Drop rows where target missing
    df = df.dropna(subset=[TARGET_COL]).copy()

    # Split
    train_df, test_df = time_based_split(df, TEST_FRACTION)
    print(f"Time range: {df[TIME_COL].min()} -> {df[TIME_COL].max()}")
    print(f"Rows: {len(df)} Unique parkplaetze: {df[ID_COL].nunique()}")
    print(f"Train rows: {len(train_df)} Test rows: {len(test_df)}")

    results_rows = []

    # Define model runs
    runs = [
        ("RF_full", "rf", True),
        ("RF_no_lag1h", "rf", False),
    ]
    if XGB_AVAILABLE:
        runs.append(("XGB_full", "xgb", True))
    else:
        print("xgboost not available -> skipping XGB_full.")

    for model_name, model_type, include_lag1h in runs:
        print("\n==============================")
        print(f"Running: {model_name}")
        print("==============================")

        model_dir = out_root / model_name
        ensure_dir(model_dir)

        feature_names = get_feature_list(include_lag1h=include_lag1h)

        # Prepare X/y
        X_train_df = train_df[feature_names].copy()
        y_train = train_df[TARGET_COL].astype(float).values
        X_test_df = test_df[feature_names].copy()
        y_test = test_df[TARGET_COL].astype(float).values

        # Impute missing with train medians
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train_df)
        X_test = imputer.transform(X_test_df)

        # Train
        t0 = time.time()
        if model_type == "rf":
            model = train_rf(X_train, y_train)
        elif model_type == "xgb":
            model = train_xgb(X_train, y_train)
        else:
            raise ValueError("Unknown model_type")
        train_time = time.time() - t0

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        m_train = compute_metrics(y_train, y_pred_train)
        m_test = compute_metrics(y_test, y_pred_test)

        metrics = {
            "model_name": model_name,
            "model_type": model_type,
            "include_lag1h": include_lag1h,
            "train_time_sec": float(train_time),
            "train": m_train,
            "test": m_test,
            "feature_count": int(len(feature_names)),
            "feature_names": feature_names
        }
        save_json(metrics, model_dir / "metrics.json")

        print("TEST metrics:", m_test)

        # Save model + imputer
        joblib.dump({"model": model, "imputer": imputer, "features": feature_names}, model_dir / f"{model_name}.joblib")

        # Feature importance
        fi = get_feature_importance(model, feature_names)
        fi.to_csv(model_dir / "feature_importance.csv", index=False)

        # Plot feature importance
        top = fi.head(20).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Importance")
        plt.title(f"Feature Importance (Top 20) - {model_name}")
        plot_and_save(model_dir / "feature_importance_top20.png")

        # Prediction dataframe for time plots
        pred_df = test_df[[TIME_COL, ID_COL, TARGET_COL]].copy()
        pred_df["y_pred"] = y_pred_test
        pred_df["resid"] = pred_df["y_pred"] - pred_df[TARGET_COL]
        # Save a small sample to keep file sizes sane
        pred_sample = sample_df(pred_df, min(200_000, len(pred_df)))
        pred_sample.to_csv(model_dir / "predictions_test_SAMPLE.csv", index=False)

        # Plots
        make_common_plots(
            out_dir=model_dir,
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred_test,
            pred_df_for_time=pred_df[[TIME_COL, "resid"]]
        )

        # SHAP only for selected model
        if DO_SHAP and (model_name == SHAP_MODEL_NAME):
            if not SHAP_AVAILABLE:
                print("SHAP not installed -> skipping.")
            else:
                print(f"Running SHAP for {model_name} with {SHAP_SAMPLES} samples...")
                # Sample from TEST to explain generalization behavior
                X_shap_df = test_df[feature_names].copy()
                X_shap_df = sample_df(X_shap_df, SHAP_SAMPLES)
                X_shap = imputer.transform(X_shap_df)
                run_shap(model_dir, model_name, model, X_shap, feature_names)

        # Store summary for comparison table
        results_rows.append({
            "model": model_name,
            "type": model_type,
            "features": len(feature_names),
            "include_lag1h": include_lag1h,
            "train_time_sec": train_time,
            "MAE": m_test["MAE"],
            "RMSE": m_test["RMSE"],
            "R2": m_test["R2"],
            "MAPE_percent": m_test["MAPE_percent"],
            "MedianAE": m_test["MedianAE"],
            "P90AE": m_test["P90AE"],
            "N_test": m_test["N"],
        })

    # Comparison table
    comp = pd.DataFrame(results_rows).sort_values(["MAE", "RMSE"])
    comp.to_csv(out_root / "model_comparison.csv", index=False)

    # Nice comparison plot (MAE + RMSE)
    plt.figure(figsize=(9, 5))
    x = np.arange(len(comp))
    plt.bar(x - 0.2, comp["MAE"], width=0.4, label="MAE")
    plt.bar(x + 0.2, comp["RMSE"], width=0.4, label="RMSE")
    plt.xticks(x, comp["model"], rotation=20, ha="right")
    plt.ylabel("Error")
    plt.title("Model Comparison (Test)")
    plt.legend()
    plot_and_save(out_root / "model_comparison.png")

    print("\nDONE. Results saved to:", out_root)
    print("Key files:")
    print("-", (out_root / "model_comparison.csv"))
    print("-", (out_root / "model_comparison.png"))


if __name__ == "__main__":
    main()
