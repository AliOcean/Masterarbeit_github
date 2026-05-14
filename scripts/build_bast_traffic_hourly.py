# build_bast_traffic_hourly.py
import re
import zipfile
import datetime as dt
import pandas as pd

ZIP_PATHS = [
    r"DZ_2025_11_Rohdaten.zip",
    r"DZ_2025_12_Rohdaten.zip",
]

OUT_TRAFFIC_CSV = "bast_traffic_hourly_2025-11_2025-12.csv"
OUT_META_CSV = "bast_station_metadata_2025-11_2025-12.csv"


def _read_meta_from_zip(z: zipfile.ZipFile) -> pd.DataFrame:
    meta_name = next(n for n in z.namelist() if "Metadaten.csv" in n)
    with z.open(meta_name) as f:
        df = pd.read_csv(f, sep=";", encoding="latin1")
    return df


def _station_id_from_filename(name: str) -> int | None:
    # Beispiele: ".../HE6105.25c" oder ".../BB3601.25B"
    m = re.search(r"/[A-Z]{2}(\d+)\.25", name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_station_file(text: str, station_id: int) -> pd.DataFrame:
    """
    Korrigierter Parser:
    - BASt Zeilen können mehr als 2*k Werte enthalten (mehrere Fahrstreifen/Blöcke)
    - Wir summieren alle Gruppen (len(vals)//k) auf.
    """
    lines = text.splitlines()

    # Klassenzeile, z.B.: "S02 09 KFZ SV Mot Pkw ...;"
    sline = next((ln for ln in lines if ln.startswith("S")), None)
    if sline is None:
        return pd.DataFrame(columns=["timestamp", "station_id", "kfz_total", "sv_total"])

    classes = sline.strip().rstrip(";").split()[2:]  # skip Sxx and nn
    k = len(classes)
    if k < 1:
        return pd.DataFrame(columns=["timestamp", "station_id", "kfz_total", "sv_total"])

    out = []

    for ln in lines:
        # Datenzeilen: yymmddiHH:MM ...
        if not re.match(r"^\d{6}i\d{2}:\d{2}", ln):
            continue

        parts = ln.split()
        ts = parts[0]  # yymmddiHH:MM

        y = int(ts[0:2])
        m = int(ts[2:4])
        d = int(ts[4:6])
        hh = int(ts[7:9])
        mm = int(ts[10:12])
        year = 2000 + y

        # Sonderfall: 24:00 -> nächster Tag 00:00
        if hh == 24 and mm == 0:
            timestamp = dt.datetime(year, m, d, 0, 0) + dt.timedelta(days=1)
        else:
            timestamp = dt.datetime(year, m, d, hh, mm)

        # Werte wie "13d" -> 13 (alle Nicht-Ziffern entfernen)
        vals = []
        for p in parts[1:]:
            num = re.sub(r"\D", "", p)
            vals.append(int(num) if num else 0)

        if len(vals) < k:
            continue

        groups = len(vals) // k
        if groups < 1:
            continue

        kfz_total = 0
        sv_total = 0

        # BASt: Klasse 0 = KFZ, Klasse 1 = SV (wenn vorhanden)
        for g in range(groups):
            offset = g * k
            kfz_total += vals[offset + 0]
            if k >= 2:
                sv_total += vals[offset + 1]

        out.append((timestamp, station_id, kfz_total, sv_total))

    return pd.DataFrame(out, columns=["timestamp", "station_id", "kfz_total", "sv_total"])


def build():
    traffic_frames = []
    meta_frames = []

    for zip_path in ZIP_PATHS:
        with zipfile.ZipFile(zip_path) as z:
            # Metadaten
            meta = _read_meta_from_zip(z)
            meta["source_zip"] = zip_path
            meta_frames.append(meta)

            # Station files (alles außer Metadaten)
            for name in z.namelist():
                if "Metadaten.csv" in name:
                    continue
                if not re.search(r"\.25[bBcC]$", name):
                    continue

                station_id = _station_id_from_filename(name)
                if station_id is None:
                    continue

                text = z.read(name).decode("latin1", errors="ignore")
                df_station = _parse_station_file(text, station_id)
                if not df_station.empty:
                    traffic_frames.append(df_station)

    traffic = pd.concat(traffic_frames, ignore_index=True)
    traffic = traffic.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    meta_all = pd.concat(meta_frames, ignore_index=True)
    # station_id heißt in Metadaten typischerweise "Dauerzaehlstellennummer"
    if "Dauerzaehlstellennummer" in meta_all.columns:
        meta_all = meta_all.drop_duplicates(subset=["Dauerzaehlstellennummer"]).reset_index(drop=True)

    traffic.to_csv(OUT_TRAFFIC_CSV, index=False)
    meta_all.to_csv(OUT_META_CSV, index=False, sep=";")

    print("Wrote:", OUT_TRAFFIC_CSV, "rows=", len(traffic))
    print("Wrote:", OUT_META_CSV, "rows=", len(meta_all))


if __name__ == "__main__":
    build()
