# dwd_hourly_recent_multi.py (Version 2.2)
# Excel:
#  - "Stationen": gültige Metadaten (aus RR-Stationstabelle)
#  - "Messungen": stündlich, JOIN über stations_id + datetime, inkl. RR + weitere Parameter
#  - "Coverage": Abdeckung (%) pro Station & Variable (vor Backfill)
# Robust: HTTPS bevorzugt, HTTP-Fallback, Retries, Parallel, tolerante Spaltenerkennung
# NEU 2.1: Konsolidierung => genau 1 Zeile pro (stations_id, datetime)
# NEU 2.2: Coverage-Sheet + Spatial Backfill (Nachbarstationen) für fehlende Werte

import io
import math
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests.exceptions import SSLError, ConnectionError as ReqConnError, ReadTimeout, HTTPError
import pandas as pd
import requests

# ---------------- Einstellungen ----------------
FORCE_HTTP = False          # True => HTTP erzwingen
VERIFY_TLS = True
MAX_STATIONS = 10           # 0 = alle Stationen/Zips pro Parameter
MAX_WORKERS = 8
SKIP_MEASUREMENTS = False   # True => nur Stationen-Sheet

TIMEOUT_CONNECT = 8
TIMEOUT_READ = 25
DEBUG = True
MEAS_DEBUG = True

BASE_ROOT = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/"


def base_url(path: str) -> str:
    url = BASE_ROOT + path
    return url if not FORCE_HTTP else url.replace("https://", "http://")


# Zeitfenster / Output
FROM = pd.Timestamp("2025-05-01 00:00")
TO   = pd.Timestamp("2025-05-07 23:00")
OUT_XLSX = "DWD_hourly_recent_MayWeek2025.xlsx"

# ---- Parameter-Katalog ----
PARAMS = {
    "precipitation": dict(
        code="RR", recent="precipitation/recent/",
        out_cols={"rr_mm": "rr_mm", "rr_spezial_indikator": "rr_spezial_indikator", "qualitaet": "qualitaet"},
        value_regex=[r"^R1$", r"^RS$", r"^R\d+"],
        extra_map={"RS_IND": "rr_spezial_indikator", "QN_8": "qualitaet"}
    ),
    "air_temperature": dict(
        code="TU", recent="air_temperature/recent/",
        out_cols={"temp_air_c": "temp_air_c"},
        value_regex=[r"^TT", r"^T(?!F|D)"], extra_map={}
    ),
    "dew_point": dict(
        code="TD", recent="dew_point/recent/",
        out_cols={"dew_point_c": "dew_point_c"},
        value_regex=[r"^TD"], extra_map={}
    ),
    "moisture": dict(
        code="TF", recent="moisture/recent/",
        out_cols={"rel_humidity_pct": "rel_humidity_pct"},
        value_regex=[r"^RF", r"^U"], extra_map={}
    ),
    "wind": dict(
        code="FF", recent="wind/recent/",
        out_cols={"wind_speed_ms": "wind_speed_ms", "wind_dir_deg": "wind_dir_deg"},
        value_regex=[r"^F[^A-Z]?$", r"^FF"],  # Geschwindigkeit
        dir_regex=[r"^D[^A-Z]?$", r"^DD"],    # Richtung
        extra_map={}
    ),
    "pressure": dict(
        code="P0", recent="pressure/recent/",
        out_cols={"pressure_hpa": "pressure_hpa"},
        value_regex=[r"^P0", r"^P\D"], extra_map={}
    ),
    "sun": dict(
        code="SD", recent="sun/recent/",
        out_cols={"sunshine_min": "sunshine_min"},
        value_regex=[r"^SD"], extra_map={}
    ),
    "solar": dict(
        code="ST", recent="solar/recent/",
        out_cols={"solar_radiation_wm2": "solar_radiation_wm2"},
        value_regex=[r"^GS", r"^ST"], extra_map={}
    ),
    "cloudiness": dict(
        code="N", recent="cloudiness/recent/",
        out_cols={"cloud_cover_oktas": "cloud_cover_oktas"},
        value_regex=[r"^N$"], extra_map={}
    ),
}

ENABLED_PARAMS = [
    "precipitation",
    "air_temperature",
    "dew_point",
    "moisture",
    "wind",
    "pressure",
    "sun",
    "solar",
    "cloudiness",
]

# Spalten, die wir mit Nachbarn füllen wollen (RR/Qualität lassen wir wie sie sind)
BACKFILL_COLS = [
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

# Maximaler Radius für Nachbar-Backfill (km)
BACKFILL_MAX_KM = 30.0
BACKFILL_K      = 3   # bis zu 3 nächste Nachbarn

# ---------------- HTTP-Session ----------------
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; Masterarbeit-Downloader/2.2)"}))
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.6,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=40)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = make_session()


def dbg(msg: str):
    if DEBUG:
        print(msg, flush=True)


def mdbg(msg: str):
    if MEAS_DEBUG:
        print(msg, flush=True)


def fetch_bytes(url: str) -> bytes:
    dbg(f"[HTTP] GET {url}")
    try:
        r = SESSION.get(url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), verify=VERIFY_TLS)
        r.raise_for_status()
        dbg(f"[HTTP] OK  {url} ({len(r.content)} bytes)")
        return r.content
    except (SSLError, ReqConnError, ReadTimeout, HTTPError) as e:
        dbg(f"[HTTP] WARN {e.__class__.__name__} -> Fallback, wenn https: {url}")
        if url.startswith("https://"):
            alt = url.replace("https://", "http://")
            r2 = SESSION.get(alt, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), verify=False)
            r2.raise_for_status()
            dbg(f"[HTTP] OK  {alt} ({len(r2.content)} bytes)")
            return r2.content
        raise


def fetch_text(url: str) -> str:
    data = fetch_bytes(url)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


# ---------------- ZIP/Helfer ----------------
def list_recent_zips_for(recent_url: str, code_prefix: str) -> list[str]:
    idx = fetch_text(recent_url)
    pat = rf'href\s*=\s*["\']?(stundenwerte_{re.escape(code_prefix)}_\d{{5}}_akt\.zip)["\']?'
    zips = re.findall(pat, idx, flags=re.I)
    dbg(f"[LIST] {code_prefix}: {len(zips)} ZIPs gefunden.")
    return zips


def read_txt_from_zip(zb: bytes, pattern: str) -> tuple[str | None, str | None]:
    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        for name in zf.namelist():
            if re.search(pattern, name, flags=re.I):
                with zf.open(name) as f:
                    raw = f.read()
                    try:
                        return name, raw.decode("utf-8")
                    except UnicodeDecodeError:
                        return name, raw.decode("latin-1", errors="replace")
    return None, None


# ---------------- Metadaten (robuster Tail-Split) ----------------
BUNDESLAENDER = [
    "Baden-Württemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thüringen",
]
_BUND_PATTERN = "(?:" + "|".join(map(re.escape, BUNDESLAENDER)) + ")"
BUND_SEARCH = re.compile(_BUND_PATTERN)

HEAD_RE = re.compile(
    r"^\s*(?P<stations_id>\d{5})\s+"
    r"(?P<von_datum>\d{8})\s+"
    r"(?P<bis_datum>\d{8})\s+"
    r"(?P<stationshoehe_m>-?\d+)\s+"
    r"(?P<breite>[+-]?\d+\.\d+)\s+"
    r"(?P<laenge>[+-]?\d+\.\d+)\s+"
    r"(?P<tail>.*\S)\s*$"
)

META_URL_RR = base_url("precipitation/recent/RR_Stundenwerte_Beschreibung_Stationen.txt")


def load_station_metadata() -> pd.DataFrame:
    print("1) Lade Stationsliste …", flush=True)
    dbg("Checkpoint 1: Lade META_URL (RR)")
    txt = fetch_text(META_URL_RR.replace("http://", "https://"))
    if "<html" in txt.lower() or "<!doctype" in txt.lower():
        dbg("Checkpoint 1b: HTTPS lieferte HTML – Fallback auf HTTP")
        txt = fetch_text(META_URL_RR.replace("https://", "http://"))

    dbg("Checkpoint 2: Filtere Zeilen")
    lines = [ln.rstrip("\n") for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 3:
        print(" [Warnung] Stationsdatei leer/zu kurz.", flush=True)
        return _empty_station_df()

    if ";" in lines[0]:
        dbg("Checkpoint 2b: CSV-Header erkannt")
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep=";", dtype=str, engine="python")
        return _normalize_and_filter_station_df(df)

    data_lines = lines[2:]
    n = len(data_lines)
    dbg(f"Checkpoint 3: Parsen der Datenzeilen (n={n})")

    recs, bad_head, bad_tail = [], 0, 0
    for idx, ln in enumerate(data_lines, 1):
        m = HEAD_RE.match(ln)
        if not m:
            bad_head += 1
            if bad_head <= 3:
                dbg(f"[Parse/HEAD] skip line {idx}: {ln[:120]}…")
            continue

        tail = m.group("tail")
        mb = BUND_SEARCH.search(tail)
        if not mb:
            bad_tail += 1
            if bad_tail <= 3:
                dbg(f"[Parse/TAIL] kein Bundesland gefunden (line {idx}): {tail[:120]}…")
            continue

        stationsname = tail[:mb.start()].strip()
        bundesland   = mb.group(0).strip()
        if not stationsname:
            bad_tail += 1
            if bad_tail <= 3:
                dbg(f"[Parse/TAIL] leerer Stationsname (line {idx}): {tail[:120]}…")
            continue

        recs.append(
            {
                "stations_id": m.group("stations_id"),
                "stationsname": stationsname,
                "bundesland": bundesland,
                "stationshoehe_m": m.group("stationshoehe_m"),
                "breite": m.group("breite"),
                "laenge": m.group("laenge"),
                "von_datum": m.group("von_datum"),
                "bis_datum": m.group("bis_datum"),
            }
        )

        if idx % 200 == 0:
            dbg(f"[Parse] Fortschritt: {idx}/{n}")

    dbg(f"Checkpoint 4: Records OK={len(recs)}  BAD_HEAD={bad_head}  BAD_TAIL={bad_tail}")
    if not recs:
        print(" [Fehler] Keine Zeile passte auf das Muster – Format geändert?", flush=True)
        return _empty_station_df()

    df = pd.DataFrame(recs).astype(str)
    return _normalize_and_filter_station_df(df)


def _empty_station_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "stations_id",
            "stationsname",
            "bundesland",
            "stationshoehe_m",
            "breite",
            "laenge",
            "gueltig_von",
            "gueltig_bis",
        ]
    )


def _normalize_and_filter_station_df(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c: str) -> str:
        c = (c or "").strip()
        c = re.sub(r"[^0-9A-Za-zÄÖÜäöüß]+", "_", c)
        return c.strip("_").lower()

    df.columns = [norm(str(c)) for c in df.columns]

    variants = {
        "stations_id": {"stations_id", "stationsid", "station_id", "stations_nr"},
        "stationsname": {"stationsname", "stations_name", "name"},
        "bundesland": {"bundesland"},
        "stationshoehe_m": {
            "stationshoehe",
            "stationshoehe_m",
            "stationshoehe_in_m",
            "stations_hoehe",
            "hoehe",
        },
        "breite": {"geobreite", "geo_breite", "geogr_breite", "breite"},
        "laenge": {"geolaenge", "geo_laenge", "geogr_laenge", "laenge", "länge"},
        "von_datum": {"von_datum", "von", "gueltig_von", "gültig_von"},
        "bis_datum": {"bis_datum", "bis", "gueltig_bis", "gültig_bis"},
    }

    def pick(varset: set[str]) -> str | None:
        for v in varset:
            if v in df.columns:
                return v
        return None

    select_map = {}
    for tgt, vs in variants.items():
        src = pick(vs)
        if src:
            select_map[src] = tgt

    if not select_map:
        print(" [Warnung] Keine erkannten Spalten. Gefunden:", df.columns.tolist(), flush=True)
        return _empty_station_df()

    out = df[list(select_map.keys())].rename(columns=select_map).copy()

    if "stations_id" in out.columns:
        out["stations_id"] = (
            out["stations_id"]
            .astype(str)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(5)
        )

    def to_dt8(x):
        s = (str(x) if pd.notna(x) else "").strip()
        s = re.sub(r"\D", "", s)[:8]
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce") if s else pd.NaT

    out["von_dt"] = out["von_datum"].map(to_dt8) if "von_datum" in out else pd.NaT
    out["bis_dt"] = out["bis_datum"].map(to_dt8) if "bis_datum" in out else pd.NaT

    start_day, end_day = FROM.normalize(), TO.normalize()
    mask = (out["von_dt"].isna() | (out["von_dt"] <= end_day)) & (
        out["bis_dt"].isna() | (out["bis_dt"] >= start_day)
    )
    out = out[mask].copy()
    out.rename(columns={"von_datum": "gueltig_von", "bis_datum": "gueltig_bis"}, inplace=True)

    if "gueltig_von" in out.columns:
        gv = pd.to_datetime(
            out["gueltig_von"]
            .astype(str)
            .str.replace(r"\D", "", regex=True)
            .str[:8],
            format="%Y%m%d",
            errors="coerce",
        )
        out = (
            out.assign(_gv=gv)
            .sort_values(["stations_id", "_gv"])
            .groupby("stations_id", as_index=False)
            .tail(1)
            .drop(columns=["_gv"])
        )

    want = [
        "stations_id",
        "stationsname",
        "bundesland",
        "stationshoehe_m",
        "breite",
        "laenge",
        "gueltig_von",
        "gueltig_bis",
    ]
    for c in want:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[want].drop_duplicates()
    print(f" Stationen (gültig um {FROM.date()}): {len(out)}", flush=True)
    return out


# ---------------- Parser generisch + Konsolidierung ----------------
PROD_TXT_PATTERN = r"produkt[_-]?.*stunde_.*\.txt$"


def _pick_value_column(df: pd.DataFrame, regex_list: list[str], exclude=set()):
    cols = list(df.columns)
    for rgx in (regex_list or []):
        for c in cols:
            if re.match(rgx, str(c).strip(), flags=re.I):
                if c not in exclude:
                    return c
    skip = {c for c in cols if re.match(r"^QN_\d+$", c)} | set(exclude) | {
        "STATIONS_ID",
        "MESS_DATUM",
        "eor",
        "RS_IND",
    }
    for c in cols:
        if c in skip:
            continue
        series = df[c]
        try:
            pd.to_numeric(series.str.replace(",", ".", regex=False), errors="raise")
            return c
        except Exception:
            continue
    return None


def _first_notna(series: pd.Series):
    for v in series:
        if pd.notna(v):
            return v
    return pd.NA


def consolidate_by_key(df: pd.DataFrame, keys=("stations_id", "datetime")) -> pd.DataFrame:
    """Sichert Eindeutigkeit je Key-Kombination. Nimmt pro Spalte das erste nicht-NA."""
    if df.empty:
        return df
    sort_cols = [c for c in keys if c in df.columns]
    df = df.sort_values(sort_cols)
    agg_map = {c: _first_notna for c in df.columns if c not in keys}
    dfc = df.groupby(list(keys), as_index=False).agg(agg_map).sort_values(list(keys))
    return dfc


def parse_measurements_generic(zb: bytes, param_cfg: dict) -> pd.DataFrame:
    fname, txt = read_txt_from_zip(zb, PROD_TXT_PATTERN)
    if not txt:
        mdbg(f"[MEAS/{param_cfg['code']}] Kein produkt_*_stunde_*.txt im ZIP.")
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(txt), sep=";", dtype=str, comment="#", engine="python")
    df.columns = [c.strip() for c in df.columns]

    if "STATIONS_ID" not in df.columns or "MESS_DATUM" not in df.columns:
        mdbg(f"[MEAS/{param_cfg['code']}] STATIONS_ID oder MESS_DATUM fehlt – unbekanntes Format.")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(
        df["MESS_DATUM"].astype(str), format="%Y%m%d%H", errors="coerce"
    )
    df = df[(df["datetime"] >= FROM) & (df["datetime"] <= TO)].copy()
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "stations_id": df["STATIONS_ID"]
            .astype(str)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(5),
            "datetime": df["datetime"],
        }
    )

    code = param_cfg["code"]
    if code == "FF":
        vcol = _pick_value_column(df, param_cfg.get("value_regex", []))
        dcol = _pick_value_column(
            df, param_cfg.get("dir_regex", []), exclude={vcol} if vcol else set()
        )
        out["wind_speed_ms"] = (
            pd.to_numeric(df[vcol].str.replace(",", ".", regex=False), errors="coerce")
            if vcol
            else pd.NA
        )
        out["wind_dir_deg"] = (
            pd.to_numeric(df[dcol].str.replace(",", ".", regex=False), errors="coerce")
            if dcol
            else pd.NA
        )
    else:
        vcol = _pick_value_column(df, param_cfg.get("value_regex", []))
        key = next(iter(param_cfg["out_cols"].keys()))
        out[key] = (
            pd.to_numeric(df[vcol].str.replace(",", ".", regex=False), errors="coerce")
            if vcol
            else pd.NA
        )

    for src, tgt in param_cfg.get("extra_map", {}).items():
        if src in df.columns:
            out[tgt] = df[src].astype(str).str.strip()

    out = consolidate_by_key(out, keys=("stations_id", "datetime"))

    mdbg(f"[MEAS/{code}] Beispiel:\n" + out.head(3).to_string(index=False))
    return out


def download_param_and_parse(zip_name: str, recent_url: str, code_prefix: str, param_cfg: dict) -> pd.DataFrame:
    url = recent_url + zip_name
    mdbg(f"[MEAS/{code_prefix}] Lade ZIP: {url}")
    zb = fetch_bytes(url)
    return parse_measurements_generic(zb, param_cfg)


# ---------------- Coverage & Backfill ----------------
def compute_coverage(wide: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Coverage pro Station & Spalte (vor Backfill)."""
    if wide.empty:
        return pd.DataFrame(columns=["stations_id", "n_rows"] + [f"cov_{c}" for c in value_cols])

    df = wide.copy()
    df["stations_id"] = df["stations_id"].astype(str)

    grp = df.groupby("stations_id")
    rows = grp["datetime"].count().rename("n_rows")

    cov_frames = [rows]
    for c in value_cols:
        if c in df.columns:
            cov = grp[c].apply(lambda s: s.notna().mean()).rename(f"cov_{c}")
            cov_frames.append(cov)

    cov_df = pd.concat(cov_frames, axis=1).reset_index()
    return cov_df


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Entfernung zwischen zwei Punkten auf der Erde in km."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def compute_neighbors(stations_df: pd.DataFrame, max_km: float, k: int) -> dict:
    """Berechnet für jede Station bis zu k Nachbarn im Umkreis max_km."""
    if stations_df.empty:
        return {}

    s = stations_df.copy()
    s["lat"] = s["breite"].astype(float)
    s["lon"] = s["laenge"].astype(float)

    ids = s["stations_id"].tolist()
    lats = s["lat"].tolist()
    lons = s["lon"].tolist()

    neighbors = {}
    n = len(ids)
    for i in range(n):
        sid = ids[i]
        lat1, lon1 = lats[i], lons[i]
        dists = []
        for j in range(n):
            if i == j:
                continue
            dist = haversine_km(lat1, lon1, lats[j], lons[j])
            if dist <= max_km:
                dists.append((ids[j], dist))
        dists.sort(key=lambda x: x[1])
        neighbors[sid] = [st for st, d in dists[:k]]
    return neighbors


def backfill_with_neighbors(
    wide: pd.DataFrame,
    neighbors: dict,
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Füllt fehlende Werte mit Werten von Nachbarstationen zur selben datetime.
    Strategie: für jede Station, jede Spalte, jede fehlende Zeile:
      - suche nacheinander in den Nachbarn, ob dort zur selben datetime ein Wert vorliegt.
      - nimm den ersten gefundenen.
    """

    if wide.empty or not value_cols:
        return wide

    df = wide.copy()
    df["stations_id"] = df["stations_id"].astype(str)

    # Flags: war der Wert ursprünglich missing?
    for c in value_cols:
        if c in df.columns:
            df[f"{c}_orig_missing"] = df[c].isna()

    # Für Performance: Indexe bauen
    df.set_index(["stations_id", "datetime"], inplace=True)

    # Hilfsfunktion: Wert aus Nachbarn holen
    def get_neighbor_value(station_id: str, dt, col: str):
        neighs = neighbors.get(station_id, [])
        for nid in neighs:
            key = (nid, dt)
            if key in df.index:
                val = df.at[key, col]
                if pd.notna(val):
                    return val
        return pd.NA

    # über alle Spalten iterieren
    for col in value_cols:
        if col not in df.columns:
            continue
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue

        idx_missing = df.index[missing_mask]
        mdbg(f"[BACKFILL] Spalte {col}: {len(idx_missing)} fehlende Werte vor Backfill")

        filled = 0
        for sid, dt in idx_missing:
            val = get_neighbor_value(sid, dt, col)
            if pd.notna(val):
                df.at[(sid, dt), col] = val
                filled += 1

        mdbg(f"[BACKFILL] Spalte {col}: {filled} Werte mit Nachbarn gefüllt")

    df.reset_index(inplace=True)
    return df


# ---------------- main ----------------
def main():
    dbg("=== START ===")
    stations_df = load_station_metadata()

    if SKIP_MEASUREMENTS:
        print("2) Messungs-Download deaktiviert.", flush=True)
        wide = pd.DataFrame(columns=["stations_id", "datetime"])
        coverage = compute_coverage(wide, BACKFILL_COLS)
        errors_total = 0
    else:
        print("2) Sammle Messungen über mehrere Parameter …", flush=True)

        param_zip_map = {}
        for p in ENABLED_PARAMS:
            cfg = PARAMS[p]
            recent_url = base_url(cfg["recent"])
            zlist = list_recent_zips_for(recent_url, cfg["code"])
            if MAX_STATIONS > 0:
                zlist = zlist[:MAX_STATIONS]
                print(f"   {cfg['code']}: Testmodus – {len(zlist)} ZIPs (erstes Segment).", flush=True)
            param_zip_map[p] = (recent_url, cfg, zlist)

        all_frames = {}
        errors_total = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {}
            for p, (recent_url, cfg, zips) in param_zip_map.items():
                for name in zips:
                    futures[
                        ex.submit(download_param_and_parse, name, recent_url, cfg["code"], cfg)
                    ] = (p, name, cfg)

            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                p, name, cfg = futures[fut]
                try:
                    dfm = fut.result()
                    if not dfm.empty:
                        all_frames.setdefault(p, []).append(dfm)
                except Exception as e:
                    errors_total += 1
                    print(f"[Warnung] {cfg['code']}/{name}: {e}", flush=True)
                done += 1
                if done % 25 == 0 or done == total:
                    print(f" … verarbeitet: {done}/{total}", flush=True)

        merged = None
        for p in ENABLED_PARAMS:
            frames = all_frames.get(p, [])
            if not frames:
                continue
            dfp = pd.concat(frames, ignore_index=True)
            dfp = consolidate_by_key(dfp, keys=("stations_id", "datetime"))
            if merged is None:
                merged = dfp
            else:
                merged = pd.merge(
                    merged, dfp, on=["stations_id", "datetime"], how="outer"
                )

        wide = merged if merged is not None else pd.DataFrame(
            columns=["stations_id", "datetime"]
        )
        wide = consolidate_by_key(wide, keys=("stations_id", "datetime"))

        # Coverage vor Backfill
        coverage = compute_coverage(wide, BACKFILL_COLS)

        # Spatial Backfill
        print("2b) Spatial Backfill mit Nachbarstationen …", flush=True)
        neighbors = compute_neighbors(stations_df, BACKFILL_MAX_KM, BACKFILL_K)
        wide = backfill_with_neighbors(wide, neighbors, BACKFILL_COLS)

    print("3) Schreibe Excel …", flush=True)
    outfile = OUT_XLSX
    try:
        writer = pd.ExcelWriter(
            outfile, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm"
        )
    except Exception:
        try:
            writer = pd.ExcelWriter(
                outfile, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm"
            )
        except Exception as e2:
            if isinstance(e2, PermissionError) or "Permission denied" in str(e2):
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                outfile = f"DWD_hourly_recent_MayWeek2025_{ts}.xlsx"
                print(
                    f" [Hinweis] {OUT_XLSX} gesperrt – schreibe stattdessen: {outfile}",
                    flush=True,
                )
                writer = pd.ExcelWriter(
                    outfile, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm"
                )
            else:
                raise

    with writer as xl:
        stations_df.to_excel(xl, sheet_name="Stationen", index=False)
        if not wide.empty and "datetime" in wide.columns:
            wide.sort_values(["stations_id", "datetime"], inplace=True)
        wide.to_excel(xl, sheet_name="Messungen", index=False)
        coverage.to_excel(xl, sheet_name="Coverage", index=False)

    print(f"Fertig: {outfile}", flush=True)
    n_meas = len(wide) if not wide.empty else 0
    print(
        f" Stationen: {len(stations_df)} | Zeilen Messungen (JOIN+Backfill): {n_meas} | Fehler: {errors_total}",
        flush=True,
    )
    dbg("=== ENDE ===")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
