# app.py
# Start:
#   python -m streamlit run app.py

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import xgboost as xgb

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium


# =========================
# CONFIG / CONSTANTS
# =========================
st.set_page_config(page_title="SID LKW-Parkplatz – Prognose Prototyp", layout="wide")

MODEL_PATH = "model/xgb_model.json"
FEATURES_PATH = "model/feature_list.json"
IMPUTER_PATH = "model/imputer.joblib"

DATA_PATH = "data/features_for_app.parquet"
MAPPING_PATH = "data/parkplatz_to_bast_station_mapping_enriched.csv"  # optional

PLOTS_DIR = "assets/plots"

STATE_VALUES = np.array([30.0, 75.0, 95.0], dtype=float)
SID_MAP: Dict[float, Tuple[str, str]] = {
    30.0: ("spacesAvailable", "frei"),
    75.0: ("almostFull", "fast voll"),
    95.0: ("Full", "voll"),
}

P90AE_PERCENT_POINTS = 25.0


# =========================
# DATA MODELS
# =========================
@dataclass(frozen=True)
class SidebarConfig:
    mode: str
    ts: pd.Timestamp
    map_color_mode: str
    radius_km: int
    top_k: int
    start_txt: str
    dest_txt: str
    buffer_km: int
    top_k_route: int
    horizon: int
    show_pngs: bool
    visible_states: List[str]
    search_pid: str
    show_route_overlay: bool
    show_route_candidates: bool
    show_heatmap: bool
    heatmap_mode: str


# =========================
# HELPERS
# =========================
def snap_to_state(x: float) -> float:
    if x is None or pd.isna(x):
        return np.nan
    idx = int(np.argmin(np.abs(STATE_VALUES - float(x))))
    return float(STATE_VALUES[idx])


def to_sid_status(x: float) -> Tuple[str, str]:
    v = snap_to_state(x)
    if pd.isna(v):
        return ("unknown", "unbekannt")
    return SID_MAP[v]


def free_vs_occupied(capacity: float, occ_percent: float) -> Tuple[Optional[int], Optional[int]]:
    if capacity is None or pd.isna(capacity):
        return None, None
    cap = int(round(float(capacity)))
    if cap <= 0:
        return None, None
    occ = int(round((float(occ_percent) / 100.0) * cap))
    occ = max(0, min(cap, occ))
    return cap - occ, occ


def parse_latlon(text: str) -> Tuple[float, float]:
    t = text.strip().replace(";", ",")
    parts = [p.strip() for p in t.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Bitte als 'lat, lon' eingeben, z. B. '50.11, 8.68'")
    return float(parts[0]), float(parts[1])


def try_parse_latlon(text: str) -> Optional[Tuple[float, float]]:
    try:
        return parse_latlon(text)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def geocode_place(query: str) -> Optional[Tuple[float, float, str]]:
    """
    Geocoding über Nominatim.
    Rückgabe:
    (lat, lon, display_name)
    """
    q = (query or "").strip()
    if not q:
        return None

    try:
        url = (
            "https://nominatim.openstreetmap.org/search?"
            f"q={quote(q)}&format=jsonv2&limit=1"
        )
        req = Request(
            url,
            headers={
                "User-Agent": "SID-LKW-Parkplatz-Prototyp/1.0"
            },
        )
        with urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not data:
            return None

        top = data[0]
        lat = float(top["lat"])
        lon = float(top["lon"])
        display_name = str(top.get("display_name", q))
        return lat, lon, display_name
    except Exception:
        return None


def resolve_location(text: str) -> Tuple[Tuple[float, float], str]:
    """
    Akzeptiert entweder:
    - 'lat, lon'
    - Ortsname, z. B. 'Frankfurt am Main'
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Leere Eingabe.")

    latlon = try_parse_latlon(raw)
    if latlon is not None:
        return latlon, raw

    geo = geocode_place(raw)
    if geo is None:
        raise ValueError(
            f"Ort oder Koordinaten konnten nicht aufgelöst werden: '{raw}'. "
            "Bitte entweder 'lat, lon' oder einen eindeutigen Ortsnamen eingeben."
        )

    lat, lon, label = geo
    return (lat, lon), label


def haversine_km_vec(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def point_line_distance_km_vec(lat_p, lon_p, lat_a, lon_a, lat_b, lon_b) -> np.ndarray:
    R = 6371.0
    lat0 = (lat_a + lat_b) / 2.0
    lon0 = (lon_a + lon_b) / 2.0

    def to_xy(lat, lon):
        x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
        y = np.radians(lat - lat0) * R
        return x, y

    px, py = to_xy(lat_p, lon_p)
    ax, ay = to_xy(lat_a, lon_a)
    bx, by = to_xy(lat_b, lon_b)

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    ab2 = np.where(ab2 == 0, 1e-12, ab2)

    t = (apx * abx + apy * aby) / ab2
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby

    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def latlon_to_xy_km(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    R = 6371.0
    x = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R
    return x, y


def xy_km_to_latlon(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
    R = 6371.0
    lat = lat0 + math.degrees(y / R)
    lon = lon0 + math.degrees(x / (R * math.cos(math.radians(lat0))))
    return lat, lon


def point_to_route_distance_km(
    lat_p: float,
    lon_p: float,
    route_points: List[Tuple[float, float]],
) -> Tuple[float, Tuple[float, float], float]:
    if not route_points or len(route_points) < 2:
        return np.inf, (np.nan, np.nan), np.inf

    best_dist = np.inf
    best_point = (np.nan, np.nan)
    best_progress_km = np.inf

    cumulative = 0.0

    for i in range(len(route_points) - 1):
        lat_a, lon_a = route_points[i]
        lat_b, lon_b = route_points[i + 1]

        lat0 = (lat_a + lat_b) / 2.0
        lon0 = (lon_a + lon_b) / 2.0

        ax, ay = latlon_to_xy_km(lat_a, lon_a, lat0, lon0)
        bx, by = latlon_to_xy_km(lat_b, lon_b, lat0, lon0)
        px, py = latlon_to_xy_km(lat_p, lon_p, lat0, lon0)

        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-12:
            seg_len = 0.0
            t = 0.0
        else:
            t = (apx * abx + apy * aby) / ab2
            t = max(0.0, min(1.0, t))
            seg_len = math.sqrt(ab2)

        cx = ax + t * abx
        cy = ay + t * aby

        dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        c_lat, c_lon = xy_km_to_latlon(cx, cy, lat0, lon0)

        progress = cumulative + t * seg_len

        if dist < best_dist:
            best_dist = dist
            best_point = (c_lat, c_lon)
            best_progress_km = progress

        cumulative += seg_len

    return best_dist, best_point, best_progress_km


def icon_color_for_map(mode: str, ist: Optional[float], pred: Optional[float], abs_err_pp: Optional[float]) -> str:
    def status_color(v: Optional[float]) -> str:
        if v is None or pd.isna(v):
            return "gray"
        if v <= 30:
            return "green"
        if v <= 75:
            return "orange"
        return "red"

    if mode == "Pred (Status)":
        return status_color(pred)
    if mode == "Ist (Status)":
        return status_color(ist)

    v = abs_err_pp
    if v is None or pd.isna(v):
        return "gray"
    if v < 5:
        return "green"
    if v < 15:
        return "orange"
    return "red"


def status_text(pred_state: float) -> str:
    return to_sid_status(pred_state)[1]


def recommendation_score(pred_state: float, dist_km: float) -> float:
    status_penalty = {30.0: 0.0, 75.0: 50.0, 95.0: 100.0}.get(float(pred_state), 999.0)
    return status_penalty + float(dist_km)


def route_recommendation_score(pred_state: float, dist_to_route_km: float, progress_km: float) -> float:
    status_penalty = {30.0: 0.0, 75.0: 45.0, 95.0: 90.0}.get(float(pred_state), 999.0)
    return status_penalty + 1.2 * float(dist_to_route_km) + 0.03 * float(progress_km)


def availability_weight(pred_state: float) -> float:
    v = snap_to_state(pred_state)
    if pd.isna(v):
        return 0.1
    if v == 30.0:
        return 1.0
    if v == 75.0:
        return 0.45
    return 0.10


def legend_html(map_color_mode: str) -> str:
    if map_color_mode in ["Pred (Status)", "Ist (Status)"]:
        title = "Legende: SID-Status"
        rows = [("green", "frei (30)"), ("orange", "fast voll (75)"), ("red", "voll (95)")]
    else:
        title = "Legende: Absolutfehler"
        rows = [("green", "< 5 pp"), ("orange", "5–15 pp"), ("red", "> 15 pp")]

    items = "".join(
        f"""
        <div class="sid-legend-row">
          <div class="sid-legend-swatch" style="background:{c};"></div>
          <div class="sid-legend-text">{txt}</div>
        </div>
        """
        for c, txt in rows
    )

    return f"""
    <style>
      .sid-legend {{
        position: fixed;
        bottom: 26px;
        left: 26px;
        z-index: 9999;
        background: rgba(255,255,255,0.96);
        color: #111;
        padding: 12px 14px;
        border: 1px solid #cfcfcf;
        border-radius: 10px;
        font-size: 13px;
        line-height: 1.25;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        max-width: 220px;
      }}
      .sid-legend-title {{
        font-weight: 750;
        margin-bottom: 8px;
      }}
      .sid-legend-row {{
        display:flex;
        align-items:center;
        gap:10px;
        margin: 6px 0;
      }}
      .sid-legend-swatch {{
        width: 14px;
        height: 14px;
        border: 1px solid #222;
        border-radius: 3px;
        flex: 0 0 auto;
      }}
      .sid-legend-text {{
        white-space: nowrap;
      }}
    </style>
    <div class="sid-legend">
      <div class="sid-legend-title">{title}</div>
      {items}
    </div>
    """


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = [str(int(v)) for v in STATE_VALUES.tolist()]
    mat = pd.DataFrame(0, index=labels, columns=labels)
    for t, p in zip(y_true, y_pred):
        if t in mat.index and p in mat.columns:
            mat.loc[t, p] += 1
    return mat


def classification_metrics_3state(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = [str(int(v)) for v in STATE_VALUES]
    cm = confusion_table(y_true, y_pred)

    recalls, precisions, f1s = {}, {}, {}

    for lbl in labels:
        tp = float(cm.loc[lbl, lbl])
        fn = float(cm.loc[lbl, :].sum() - tp)
        fp = float(cm.loc[:, lbl].sum() - tp)

        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan

        recalls[lbl] = rec
        precisions[lbl] = prec

        if np.isfinite(rec) and np.isfinite(prec) and (rec + prec) > 0:
            f1s[lbl] = 2.0 * prec * rec / (prec + rec)
        else:
            f1s[lbl] = np.nan

    acc = float(np.mean(y_true == y_pred)) if len(y_true) else np.nan
    bal_acc = float(np.nanmean(list(recalls.values()))) if len(y_true) else np.nan
    macro_f1 = float(np.nanmean(list(f1s.values()))) if len(y_true) else np.nan

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "recalls": recalls,
        "precisions": precisions,
        "f1s": f1s,
        "cm": cm,
    }


@st.cache_data(show_spinner=False)
def get_osrm_route(start: Tuple[float, float], dest: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
    try:
        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{start[1]},{start[0]};{dest[1]},{dest[0]}"
            "?overview=full&geometries=geojson"
        )
        req = Request(
            url,
            headers={
                "User-Agent": "SID-LKW-Parkplatz-Prototyp/1.0"
            },
        )
        with urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        routes = data.get("routes", [])
        if not routes:
            return None

        coords = routes[0]["geometry"]["coordinates"]
        route = [(float(c[1]), float(c[0])) for c in coords]
        if len(route) < 2:
            return None
        return route
    except Exception:
        return None


# =========================
# LOADERS
# =========================
@st.cache_resource
def load_model_artifacts():
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    imputer = joblib.load(IMPUTER_PATH)
    return booster, feature_cols, imputer


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    mapping = None
    try:
        mapping = pd.read_csv(MAPPING_PATH)
    except Exception:
        mapping = None

    return df, mapping


def assert_features_present(df: pd.DataFrame, feature_cols: List[str]) -> None:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        msg = "Im Dataset fehlen Feature-Spalten, die das Modell erwartet:\n\n- " + "\n- ".join(missing[:30])
        if len(missing) > 30:
            msg += f"\n… und {len(missing) - 30} weitere."
        st.error(msg)
        st.stop()


def batch_predict(df_any: pd.DataFrame, booster: xgb.Booster, feature_cols: List[str], imputer) -> np.ndarray:
    X = df_any[feature_cols].copy()
    X_imp = imputer.transform(X.values)
    dmat = xgb.DMatrix(X_imp, feature_names=feature_cols)
    preds = booster.predict(dmat).astype(float)
    return np.clip(preds, 0.0, 100.0)


def build_parking_table(df_any: pd.DataFrame, mapping: Optional[pd.DataFrame]) -> pd.DataFrame:
    cols = set(df_any.columns)
    use = df_any[["parkplatz_id"]].drop_duplicates().copy()

    if "latitude" in cols and "longitude" in cols:
        use = use.merge(
            df_any[["parkplatz_id", "latitude", "longitude"]].drop_duplicates(),
            on="parkplatz_id",
            how="left",
        )
    elif mapping is not None and {"parkplatz_id", "latitude", "longitude"}.issubset(mapping.columns):
        use = use.merge(mapping[["parkplatz_id", "latitude", "longitude"]], on="parkplatz_id", how="left")
    else:
        use["latitude"] = np.nan
        use["longitude"] = np.nan

    if "capacity" in cols:
        use = use.merge(df_any[["parkplatz_id", "capacity"]].drop_duplicates(), on="parkplatz_id", how="left")
    else:
        use["capacity"] = np.nan

    use = use.dropna(subset=["latitude", "longitude"])
    return use


# =========================
# SIDEBAR
# =========================
def render_sidebar(ts_all: pd.Series) -> SidebarConfig:
    ts_now = ts_all.max()
    ts_min = ts_all.min()

    with st.sidebar:
        st.header("Konfiguration")

        mode = st.radio("Modus", ["Backtesting (Ist vs Pred)", "Forecast (nur Pred)"], index=0)
        st.caption(f"Zeitraum: **{ts_min}** → **{ts_now}**")
        st.caption(f"„Jetzt“ (Demo) = **{ts_now}**")

        ts_options = ts_all.tail(300 if mode.startswith("Forecast") else 500).to_list()
        ts = st.selectbox("Zeitpunkt", options=ts_options, index=len(ts_options) - 1)

        st.divider()
        st.subheader("Karte")
        map_color_mode = st.selectbox(
            "Marker einfärben nach",
            ["Pred (Status)", "Ist (Status)", "Absolutfehler (pp)"],
            index=0,
        )

        visible_states = st.multiselect(
            "Sichtbare Status",
            options=["frei", "fast voll", "voll"],
            default=["frei", "fast voll", "voll"],
        )

        search_pid = st.text_input("Parkplatz-ID suchen", value="")

        show_heatmap = st.checkbox("Heatmap anzeigen", value=False)
        heatmap_mode = st.selectbox(
            "Heatmap-Modus",
            ["Verfügbarkeit", "Auslastung"],
            index=0,
            disabled=not show_heatmap,
        )

        st.divider()
        st.subheader("Alternativen")
        radius_km = st.slider("Umkreis (km)", 5, 200, 50, 5)
        top_k = st.slider("Top-K", 3, 30, 10, 1)

        st.divider()
        st.subheader("Route")
        st.caption("Eingabe entweder als 'lat, lon' oder als Ortsname, z. B. 'Frankfurt am Main'")
        start_txt = st.text_input("Start", value="Frankfurt am Main")
        dest_txt = st.text_input("Ziel", value="Mannheim")
        buffer_km = st.slider("Korridor-Breite (km)", 5, 100, 25, 5)
        top_k_route = st.slider("Top-K entlang Route", 3, 40, 12, 1)
        show_route_overlay = st.checkbox("Route auf Karte anzeigen", value=True)
        show_route_candidates = st.checkbox("Routen-Kandidaten auf Karte hervorheben", value=True)

        st.divider()
        st.subheader("Kurzfrist")
        horizon = st.slider("Horizont (Stunden)", 1, 48, 12, 1)

        st.divider()
        show_pngs = st.checkbox("PNGs anzeigen", True)

    return SidebarConfig(
        mode=mode,
        ts=ts,
        map_color_mode=map_color_mode,
        radius_km=radius_km,
        top_k=top_k,
        start_txt=start_txt,
        dest_txt=dest_txt,
        buffer_km=buffer_km,
        top_k_route=top_k_route,
        horizon=horizon,
        show_pngs=show_pngs,
        visible_states=visible_states,
        search_pid=search_pid.strip(),
        show_route_overlay=show_route_overlay,
        show_route_candidates=show_route_candidates,
        show_heatmap=show_heatmap,
        heatmap_mode=heatmap_mode,
    )


# =========================
# UI
# =========================
def render_model_card():
    with st.expander("Artefakt & Evaluation (kurz)", expanded=False):
        st.markdown(
            """
- **Artefakt:** Streamlit-Prototyp zur Entscheidungsunterstützung (Karte, Alternativen, Route, Kurzfrist).
- **Modell:** XGBoost-Regressor, Output → **SID-Zustände {30, 75, 95}**.
- **Evaluation:** Backtesting pro Timestamp + Klassifikationsmetriken.
- **Unsicherheit:** P90AE als robustes Fehlerband (± Prozentpunkte).
- **Routing:** Echte Fahrroute + Kandidaten entlang des Route-Korridors.
            """
        )


def render_snapshot(df_ts: pd.DataFrame, is_backtesting: bool) -> None:
    st.subheader("Snapshot (Timestamp)")

    pred_vc = df_ts["pred_state"].value_counts().to_dict()
    st_cols = st.columns(6)

    st_cols[0].metric("N Parkplätze", f"{len(df_ts)}")
    st_cols[1].metric("Pred frei", int(pred_vc.get(30.0, 0)))
    st_cols[2].metric("Pred fast voll", int(pred_vc.get(75.0, 0)))
    st_cols[3].metric("Pred voll", int(pred_vc.get(95.0, 0)))
    st_cols[4].metric("P90AE (Test)", f"±{P90AE_PERCENT_POINTS:.0f} pp")

    if is_backtesting:
        coverage = float(df_ts["ist_state"].notna().mean()) if "ist_state" in df_ts.columns else 0.0
        mae_t = float(df_ts["abs_error_pp"].mean())
        rmse_t = float(np.sqrt(np.mean((df_ts["pred_state"] - df_ts["ist_state"]) ** 2)))
        acc_t = float(np.mean(df_ts["pred_state"] == df_ts["ist_state"]))

        st_cols[5].metric("Accuracy@t", f"{acc_t*100:.1f}%")
        st.caption(
            f"Backtesting-Metriken: Coverage={coverage*100:.1f}% | MAE@t={mae_t:.1f} pp | RMSE@t={rmse_t:.1f} pp"
        )
    else:
        st_cols[5].metric("Modus", "Forecast")
        st.caption("Forecast: Ist-Werte werden nicht zur Bewertung genutzt.")


def render_quick_lists(df_ts: pd.DataFrame, is_backtesting: bool) -> None:
    st.subheader("Schnell-Listen")

    l1, l2, l3 = st.columns(3, gap="large")

    with l1:
        st.markdown("**Kritisch (Pred = voll)**")
        crit = df_ts[df_ts["pred_state"] == 95.0][["parkplatz_id", "pred_state"]].copy()
        if len(crit) == 0:
            st.info("Keine 'voll' Prognosen.")
        else:
            crit["Pred_Status"] = crit["pred_state"].map(lambda x: to_sid_status(x)[1])
            st.dataframe(crit[["parkplatz_id", "Pred_Status"]].head(12), use_container_width=True, height=320)

    with l2:
        if is_backtesting:
            st.markdown("**Größte Fehler (|Pred–Ist|)**")
            top_err = df_ts[["parkplatz_id", "ist_state", "pred_state", "abs_error_pp"]].copy()
            top_err = top_err.sort_values("abs_error_pp", ascending=False).head(12)
            top_err["Ist"] = top_err["ist_state"].map(lambda x: to_sid_status(x)[1])
            top_err["Pred"] = top_err["pred_state"].map(lambda x: to_sid_status(x)[1])
            top_err["|e|_pp"] = top_err["abs_error_pp"].round(1)
            st.dataframe(top_err[["parkplatz_id", "Ist", "Pred", "|e|_pp"]], use_container_width=True, height=320)
        else:
            st.markdown("**Beste Optionen (Pred = frei)**")
            free = df_ts[df_ts["pred_state"] == 30.0][["parkplatz_id", "pred_state"]].copy()
            if len(free) == 0:
                st.info("Keine 'frei' Prognosen.")
            else:
                free["Pred_Status"] = free["pred_state"].map(lambda x: to_sid_status(x)[1])
                st.dataframe(free[["parkplatz_id", "Pred_Status"]].head(12), use_container_width=True, height=320)

    with l3:
        st.markdown("**Ist-Verteilung (kompakt)**")
        if is_backtesting:
            ist_vc = df_ts["ist_state"].value_counts().to_dict()
            cA, cB, cC = st.columns(3)
            cA.metric("Ist frei", int(ist_vc.get(30.0, 0)))
            cB.metric("Ist fast voll", int(ist_vc.get(75.0, 0)))
            cC.metric("Ist voll", int(ist_vc.get(95.0, 0)))
        else:
            st.write("—")


def compute_route_candidates(
    parkings: pd.DataFrame,
    df_ts: pd.DataFrame,
    cfg: SidebarConfig,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[List[Tuple[float, float]]],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    try:
        start, start_label = resolve_location(cfg.start_txt)
        dest, dest_label = resolve_location(cfg.dest_txt)

        route_points = get_osrm_route(start, dest)
        use_real_route = route_points is not None and len(route_points) >= 2

        route_cand = parkings.merge(df_ts[["parkplatz_id", "pred_state"]], on="parkplatz_id", how="inner").copy()

        if cfg.visible_states:
            route_cand = route_cand[route_cand["pred_state"].map(status_text).isin(cfg.visible_states)].copy()

        if len(route_cand) == 0:
            return route_cand, start, dest, route_points, None, start_label, dest_label

        dist_list = []
        nearest_lat_list = []
        nearest_lon_list = []
        progress_list = []

        if use_real_route:
            for _, r in route_cand.iterrows():
                d_km, nearest_pt, progress_km = point_to_route_distance_km(
                    float(r["latitude"]),
                    float(r["longitude"]),
                    route_points,
                )
                dist_list.append(d_km)
                nearest_lat_list.append(nearest_pt[0])
                nearest_lon_list.append(nearest_pt[1])
                progress_list.append(progress_km)
        else:
            lat_p = route_cand["latitude"].astype(float).to_numpy()
            lon_p = route_cand["longitude"].astype(float).to_numpy()

            dist_line = point_line_distance_km_vec(
                lat_p,
                lon_p,
                float(start[0]),
                float(start[1]),
                float(dest[0]),
                float(dest[1]),
            )
            dist_list = dist_line.tolist()
            nearest_lat_list = [np.nan] * len(route_cand)
            nearest_lon_list = [np.nan] * len(route_cand)
            progress_list = [0.0] * len(route_cand)

        route_cand["dist_to_route_km"] = dist_list
        route_cand["nearest_route_lat"] = nearest_lat_list
        route_cand["nearest_route_lon"] = nearest_lon_list
        route_cand["progress_km"] = progress_list

        route_cand = route_cand[route_cand["dist_to_route_km"] <= cfg.buffer_km].copy()

        if len(route_cand) == 0:
            return route_cand, start, dest, route_points, None, start_label, dest_label

        route_cand["score"] = route_cand.apply(
            lambda r: route_recommendation_score(r["pred_state"], r["dist_to_route_km"], r["progress_km"]),
            axis=1,
        )
        route_cand = route_cand.sort_values(["score", "dist_to_route_km", "progress_km"]).head(cfg.top_k_route).copy()
        route_cand["route_rank"] = range(1, len(route_cand) + 1)
        return route_cand, start, dest, route_points, None, start_label, dest_label
    except Exception as e:
        return None, None, None, None, str(e), None, None


def add_heatmap_layer(m: folium.Map, map_df: pd.DataFrame, cfg: SidebarConfig) -> None:
    if len(map_df) == 0:
        return

    heat_rows = []
    for _, r in map_df.iterrows():
        pred = r.get("pred_state", np.nan)
        if pd.isna(pred):
            continue

        if cfg.heatmap_mode == "Verfügbarkeit":
            weight = availability_weight(pred)
        else:
            weight = float(np.clip(pred / 100.0, 0.05, 1.0))

        heat_rows.append([float(r["latitude"]), float(r["longitude"]), weight])

    if heat_rows:
        HeatMap(
            heat_rows,
            radius=24,
            blur=18,
            min_opacity=0.25,
        ).add_to(m)


def render_route_top3_cards(route_cand: Optional[pd.DataFrame]) -> None:
    st.markdown("**Top-3 Stopps entlang Route**")

    if route_cand is None or len(route_cand) == 0:
        st.info("Keine geeigneten Routenstopps gefunden.")
        return

    top3 = route_cand.head(3).copy()
    cols = st.columns(min(3, len(top3)))
    for i, (_, r) in enumerate(top3.iterrows()):
        with cols[i]:
            st.markdown(
                f"""
**#{int(r['route_rank'])} – {r['parkplatz_id']}**

Status: **{status_text(r['pred_state'])}**  
Abstand zur Route: **{r['dist_to_route_km']:.1f} km**  
Fortschritt auf Route: **{r['progress_km']:.1f} km**  
Score: **{r['score']:.1f}**
                """
            )


def render_map_and_details(
    df_all: pd.DataFrame,
    df_ts: pd.DataFrame,
    parkings: pd.DataFrame,
    cfg: SidebarConfig,
    is_backtesting: bool,
    booster: xgb.Booster,
    feature_cols: List[str],
    imputer,
) -> None:
    col_map, col_info = st.columns([1.8, 1.0], gap="large")

    (
        route_cand,
        route_start,
        route_dest,
        route_points,
        route_err,
        start_label,
        dest_label,
    ) = compute_route_candidates(parkings, df_ts, cfg)

    if cfg.search_pid:
        if cfg.search_pid in parkings["parkplatz_id"].astype(str).tolist():
            st.session_state.selected_pid = cfg.search_pid

    with col_map:
        st.subheader("Karte")

        if start_label and dest_label:
            st.caption(f"Route: **{start_label}** → **{dest_label}**")

        center_lat = float(parkings["latitude"].mean())
        center_lon = float(parkings["longitude"].mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, control_scale=True)

        map_df = parkings.merge(
            df_ts[["parkplatz_id", "pred_state", "ist_state", "abs_error_pp"]],
            on="parkplatz_id",
            how="left",
        )

        if cfg.visible_states:
            map_df = map_df[map_df["pred_state"].map(status_text).isin(cfg.visible_states)].copy()

        if cfg.show_heatmap:
            add_heatmap_layer(m, map_df, cfg)

        selected = st.session_state.selected_pid

        route_ids = set()
        top3_route_ids = set()
        if route_cand is not None and len(route_cand):
            route_ids = set(route_cand["parkplatz_id"].astype(str).tolist())
            top3_route_ids = set(route_cand.head(3)["parkplatz_id"].astype(str).tolist())

        if cfg.show_route_overlay and route_start and route_dest:
            folium.Marker(
                route_start,
                tooltip="Start",
                icon=folium.Icon(color="blue", icon="play", prefix="fa"),
            ).add_to(m)

            folium.Marker(
                route_dest,
                tooltip="Ziel",
                icon=folium.Icon(color="black", icon="flag-checkered", prefix="fa"),
            ).add_to(m)

            if route_points is not None and len(route_points) >= 2:
                folium.PolyLine(route_points, color="blue", weight=4, opacity=0.85).add_to(m)
            else:
                folium.PolyLine([route_start, route_dest], color="blue", weight=4, opacity=0.85).add_to(m)

        if cfg.show_route_candidates and route_cand is not None and len(route_cand) > 0:
            for _, rr in route_cand.iterrows():
                pid_r = str(rr["parkplatz_id"])
                plat = float(rr["latitude"])
                plon = float(rr["longitude"])

                if pd.notna(rr["nearest_route_lat"]) and pd.notna(rr["nearest_route_lon"]):
                    folium.PolyLine(
                        [
                            (float(rr["nearest_route_lat"]), float(rr["nearest_route_lon"])),
                            (plat, plon),
                        ],
                        color="purple",
                        weight=2 if pid_r not in top3_route_ids else 4,
                        opacity=0.6 if pid_r not in top3_route_ids else 0.9,
                        dash_array="5,6",
                    ).add_to(m)

                folium.CircleMarker(
                    location=[plat, plon],
                    radius=9 if pid_r not in top3_route_ids else 12,
                    color="purple",
                    fill=True,
                    fill_opacity=0.65,
                    tooltip=f"Route-Kandidat #{int(rr['route_rank'])}: {pid_r}",
                ).add_to(m)

        for _, r in map_df.iterrows():
            pid = str(r["parkplatz_id"])
            lat = float(r["latitude"])
            lon = float(r["longitude"])

            pred = float(r["pred_state"]) if pd.notna(r["pred_state"]) else None
            ist = float(r["ist_state"]) if pd.notna(r["ist_state"]) else None
            err = float(r["abs_error_pp"]) if pd.notna(r["abs_error_pp"]) else None

            color = icon_color_for_map(cfg.map_color_mode, ist, pred, err)

            if selected is not None and pid == selected:
                color = "blue"
            elif pid in top3_route_ids:
                color = "purple"
            elif pid in route_ids:
                color = "cadetblue"

            _, pred_label = to_sid_status(pred)
            if is_backtesting:
                _, ist_label = to_sid_status(ist)
                tt = f"{pid} | Ist: {ist_label} | Pred: {pred_label} | |e| {err:.1f} pp"
            else:
                tt = f"{pid} | Pred: {pred_label}"

            folium.Marker(
                location=[lat, lon],
                tooltip=tt,
                popup=folium.Popup(tt, max_width=420),
                icon=folium.Icon(color=color, icon="truck", prefix="fa"),
            ).add_to(m)

        m.get_root().html.add_child(folium.Element(legend_html(cfg.map_color_mode)))
        map_out = st_folium(m, height=700, use_container_width=True)

        clicked = None
        if map_out and map_out.get("last_object_clicked"):
            clicked = map_out["last_object_clicked"]
        elif map_out and map_out.get("last_clicked"):
            clicked = map_out["last_clicked"]

        if clicked:
            click_lat, click_lon = float(clicked["lat"]), float(clicked["lng"])

            lat_arr = parkings["latitude"].astype(float).to_numpy()
            lon_arr = parkings["longitude"].astype(float).to_numpy()
            d = haversine_km_vec(click_lat, click_lon, lat_arr, lon_arr)

            idx = int(np.argmin(d))
            nearest_pid = str(parkings.iloc[idx]["parkplatz_id"])
            st.session_state.selected_pid = nearest_pid
            st.success(f"Ausgewählt: {nearest_pid}")

    with col_info:
        st.subheader("Details")

        if st.session_state.selected_pid is None:
            st.info("Bitte einen Marker anklicken oder eine Parkplatz-ID suchen.")
            if not route_err:
                st.divider()
                render_route_top3_cards(route_cand)
            return

        pid = st.session_state.selected_pid
        row = df_ts[df_ts["parkplatz_id"].astype(str) == str(pid)]
        if len(row) == 0:
            st.error("Keine Daten für diesen Parkplatz beim Timestamp.")
            return

        pred_state = float(row["pred_state"].iloc[0])
        pred_code, pred_label = to_sid_status(pred_state)

        cap = row["capacity"].iloc[0] if "capacity" in row.columns else None
        free_p, occ_p = free_vs_occupied(cap, pred_state)

        d1, d2 = st.columns(2)
        d1.metric("Pred-Status", pred_label)
        d2.metric("SID-Code", pred_code)

        low = snap_to_state(pred_state - P90AE_PERCENT_POINTS)
        high = snap_to_state(pred_state + P90AE_PERCENT_POINTS)
        st.metric("Unsicherheitsband", f"{int(low)}–{int(high)}")

        with st.expander("Mehr Details", expanded=False):
            if free_p is not None:
                st.caption(
                    f"Kapazität: {int(round(float(cap)))} | frei ~ {free_p} | belegt ~ {occ_p}"
                )

            if is_backtesting:
                ist_state = float(row["ist_state"].iloc[0])
                _, ist_label = to_sid_status(ist_state)
                err = abs(pred_state - ist_state)
                x1, x2 = st.columns(2)
                x1.metric("Ist-Status", ist_label)
                x2.metric("|e| (pp)", f"{err:.1f}")

        st.divider()

        tab_reco, tab_alt, tab_route, tab_short = st.tabs(
            ["Empfehlung", "Alternativen", "Route", "Kurzfrist"]
        )

        with tab_reco:
            base = parkings.loc[parkings["parkplatz_id"].astype(str) == str(pid)].iloc[0]
            lat0, lon0 = float(base["latitude"]), float(base["longitude"])

            alts = parkings.merge(df_ts[["parkplatz_id", "pred_state"]], on="parkplatz_id", how="inner").copy()
            if cfg.visible_states:
                alts = alts[alts["pred_state"].map(status_text).isin(cfg.visible_states)].copy()

            lat_arr = alts["latitude"].astype(float).to_numpy()
            lon_arr = alts["longitude"].astype(float).to_numpy()

            d = haversine_km_vec(lat0, lon0, lat_arr, lon_arr)
            alts["dist_km"] = d
            alts = alts[(alts["dist_km"] <= cfg.radius_km) & (alts["parkplatz_id"].astype(str) != str(pid))].copy()

            if len(alts) == 0:
                st.info("Keine Empfehlung im Umkreis verfügbar.")
            else:
                alts["score"] = alts.apply(lambda r: recommendation_score(r["pred_state"], r["dist_km"]), axis=1)
                best = alts.sort_values(["score", "dist_km"]).iloc[0]

                st.success(f"Beste Alternative: {best['parkplatz_id']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Status", status_text(best["pred_state"]))
                c2.metric("Distanz", f"{best['dist_km']:.1f} km")
                c3.metric("Score", f"{best['score']:.1f}")

        with tab_alt:
            base = parkings.loc[parkings["parkplatz_id"].astype(str) == str(pid)].iloc[0]
            lat0, lon0 = float(base["latitude"]), float(base["longitude"])

            alts = parkings.merge(df_ts[["parkplatz_id", "pred_state"]], on="parkplatz_id", how="inner").copy()
            if cfg.visible_states:
                alts = alts[alts["pred_state"].map(status_text).isin(cfg.visible_states)].copy()

            lat_arr = alts["latitude"].astype(float).to_numpy()
            lon_arr = alts["longitude"].astype(float).to_numpy()

            d = haversine_km_vec(lat0, lon0, lat_arr, lon_arr)
            alts["dist_km"] = d

            alts = alts[(alts["dist_km"] <= cfg.radius_km) & (alts["parkplatz_id"].astype(str) != str(pid))].copy()

            if len(alts) == 0:
                st.info("Keine Alternativen im Umkreis.")
            else:
                alts["score"] = alts.apply(lambda r: recommendation_score(r["pred_state"], r["dist_km"]), axis=1)
                alts = alts.sort_values(["score", "dist_km"]).head(cfg.top_k)
                out = alts[["parkplatz_id", "pred_state", "dist_km", "score"]].copy()
                out["Pred_Status"] = out["pred_state"].map(lambda x: to_sid_status(x)[1])
                out["dist_km"] = out["dist_km"].round(1)
                out["score"] = out["score"].round(1)

                st.dataframe(out[["parkplatz_id", "Pred_Status", "dist_km", "score"]], use_container_width=True)

        with tab_route:
            if route_err:
                st.warning(f"Ungültige Routen-Eingabe: {route_err}")
            elif route_cand is None or len(route_cand) == 0:
                st.info("Keine Parkplätze im Route-Korridor gefunden.")
            else:
                if start_label and dest_label:
                    st.caption(f"Aufgelöste Route: **{start_label}** → **{dest_label}**")

                render_route_top3_cards(route_cand)
                st.divider()

                best_route = route_cand.iloc[0]
                st.success(f"Bester Routen-Stopp: {best_route['parkplatz_id']}")

                out = route_cand[
                    [
                        "route_rank",
                        "parkplatz_id",
                        "pred_state",
                        "dist_to_route_km",
                        "progress_km",
                        "score",
                    ]
                ].copy()
                out["Pred_Status"] = out["pred_state"].map(lambda x: to_sid_status(x)[1])
                out["dist_to_route_km"] = out["dist_to_route_km"].round(1)
                out["progress_km"] = out["progress_km"].round(1)
                out["score"] = out["score"].round(1)

                st.dataframe(
                    out[
                        [
                            "route_rank",
                            "parkplatz_id",
                            "Pred_Status",
                            "dist_to_route_km",
                            "progress_km",
                            "score",
                        ]
                    ],
                    use_container_width=True,
                )

        with tab_short:
            ts0 = df_ts["timestamp"].iloc[0]
            ts_end = ts0 + pd.Timedelta(hours=cfg.horizon)

            df_pid = df_all[df_all["parkplatz_id"].astype(str) == str(pid)].copy().sort_values("timestamp")
            df_h = df_pid[(df_pid["timestamp"] >= ts0) & (df_pid["timestamp"] <= ts_end)].copy()

            if len(df_h) == 0:
                st.info("Keine weiteren Stunden im Datensatz vorhanden.")
            else:
                df_h["pred_value"] = batch_predict(df_h, booster, feature_cols, imputer)
                df_h["pred_state"] = df_h["pred_value"].map(snap_to_state)

                chart = (
                    df_h[["timestamp", "pred_state"]]
                    .copy()
                    .set_index("timestamp")
                    .rename(columns={"pred_state": "Pred"})
                )

                if is_backtesting and "occupancy_percent" in df_h.columns:
                    df_h["ist_state"] = df_h["occupancy_percent"].map(snap_to_state)
                    chart["Ist"] = df_h["ist_state"].values
                    st.line_chart(chart[["Ist", "Pred"]])

                    tbl = df_h[["timestamp", "ist_state", "pred_state"]].copy()
                    tbl["|e|_pp"] = (tbl["pred_state"] - tbl["ist_state"]).abs().round(1)
                    tbl["Ist_Status"] = tbl["ist_state"].map(lambda x: to_sid_status(x)[1])
                    tbl["Pred_Status"] = tbl["pred_state"].map(lambda x: to_sid_status(x)[1])
                    st.dataframe(tbl[["timestamp", "Ist_Status", "Pred_Status", "|e|_pp"]], use_container_width=True)
                else:
                    st.line_chart(chart[["Pred"]])
                    tbl = df_h[["timestamp", "pred_state"]].copy()
                    tbl["Pred_Status"] = tbl["pred_state"].map(lambda x: to_sid_status(x)[1])
                    st.dataframe(tbl[["timestamp", "Pred_Status"]], use_container_width=True)


def render_classification_view(df_ts: pd.DataFrame) -> None:
    st.subheader("Klassifikations-View (3 SID-Zustände)")

    mask = df_ts["ist_state"].notna()
    if not mask.any():
        st.info("Keine Ist-Werte vorhanden → keine Klassifikationsmetriken.")
        return

    y_true = df_ts.loc[mask, "ist_state"].map(lambda x: str(int(x))).to_numpy()
    y_pred = df_ts.loc[mask, "pred_state"].map(lambda x: str(int(x))).to_numpy()

    m = classification_metrics_3state(y_true, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['acc']*100:.1f}%")
    c2.metric("Balanced Accuracy", f"{m['bal_acc']*100:.1f}%")
    c3.metric("Macro-F1", f"{m['macro_f1']*100:.1f}%")

    rec_full = m["recalls"].get("95", np.nan)
    st.caption(f"Recall(„voll“/95): {rec_full*100:.1f}%")

    st.write("**Confusion Matrix (Zeile=Ist, Spalte=Pred)**")
    st.dataframe(m["cm"], use_container_width=True)


def render_pngs() -> None:
    st.divider()
    st.subheader("Modell-Plots (PNGs)")

    if not os.path.isdir(PLOTS_DIR):
        st.info("Ordner assets/plots/ existiert noch nicht.")
        return

    pngs = [p for p in os.listdir(PLOTS_DIR) if p.lower().endswith(".png")]
    if not pngs:
        st.info("Keine PNGs in assets/plots/ gefunden.")
        return

    cols = st.columns(2)
    for i, p in enumerate(sorted(pngs)):
        with cols[i % 2]:
            st.image(os.path.join(PLOTS_DIR, p), caption=p, use_container_width=True)


# =========================
# MAIN
# =========================
def main():
    st.title("SID LKW-Parkplatz – XGBoost Prognose Prototyp")
    render_model_card()

    booster, feature_cols, imputer = load_model_artifacts()
    df, mapping = load_data()

    if "timestamp" not in df.columns:
        st.error("Dataset enthält keine Spalte 'timestamp'.")
        st.stop()

    ts_all = df["timestamp"].dropna().drop_duplicates().sort_values()
    if len(ts_all) == 0:
        st.error("Keine timestamps im Dataset.")
        st.stop()

    cfg = render_sidebar(ts_all)
    assert_features_present(df, feature_cols)

    df_ts = df[df["timestamp"] == cfg.ts].copy()
    if len(df_ts) == 0:
        st.error("Keine Zeilen für diesen Timestamp.")
        st.stop()

    df_ts["pred_value"] = batch_predict(df_ts, booster, feature_cols, imputer)
    df_ts["pred_state"] = df_ts["pred_value"].map(snap_to_state)

    is_backtesting = cfg.mode.startswith("Backtesting")
    if is_backtesting and "occupancy_percent" in df_ts.columns:
        df_ts["ist_state"] = df_ts["occupancy_percent"].map(snap_to_state)
        df_ts["abs_error_pp"] = (df_ts["pred_state"] - df_ts["ist_state"]).abs()
    else:
        df_ts["ist_state"] = np.nan
        df_ts["abs_error_pp"] = np.nan

    parkings = build_parking_table(df_ts, mapping)
    if len(parkings) == 0:
        st.error("Keine Parkplätze mit Koordinaten gefunden (latitude/longitude fehlen).")
        st.stop()

    if "selected_pid" not in st.session_state:
        st.session_state.selected_pid = None

    render_snapshot(df_ts, is_backtesting)
    render_quick_lists(df_ts, is_backtesting)

    render_map_and_details(
        df_all=df,
        df_ts=df_ts,
        parkings=parkings,
        cfg=cfg,
        is_backtesting=is_backtesting,
        booster=booster,
        feature_cols=feature_cols,
        imputer=imputer,
    )

    if is_backtesting:
        render_classification_view(df_ts)

    if cfg.show_pngs:
        render_pngs()


if __name__ == "__main__":
    main()