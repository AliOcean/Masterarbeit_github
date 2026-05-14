import geopandas as gpd

# 1) Pfad zur GeoPackage-Datei (liegt im gleichen Ordner wie dieses Script)
GPKG_PATH = "BFStr_Netz_v2025q2.gpkg"

# 2) Layernamen prüfen: In dieser Version gibt es u.a.:
#    - "BFStr_Netz_NK" = Netzknoten (Kreuzungen, Anschlussstellen, etc.)
#    - "BFStr_Netz_SK" = Netzsegmente (Abschnitte)
#    - "BFStr_Netz_NP" = Netzpunkte (weitere Punkte)
#
# Wir brauchen für die Abfahrten vor allem die Netzknoten (NK).
LAYER_NK = "BFStr_Netz_NK"

print("Lade Netzknoten (NK) aus GeoPackage...")
nk = gpd.read_file(GPKG_PATH, layer=LAYER_NK)

print("Spalten in NK:", nk.columns.tolist())
print("Anzahl aller NK:", len(nk))

# 3) Autobahn-relevante Knoten filtern:
#    - NK_BABKnoten = 'J' -> Autobahnknoten (Junction)
#    - NK_Knotenpunktfunktion = 'Anschlussstelle' -> typische Autobahnabfahrten
autobahn_as = nk[
    (nk["NK_BABKnoten"] == "J") &
    (nk["NK_Knotenpunktfunktion"] == "Anschlussstelle")
].copy()

print("Gefundene Autobahn-Anschlussstellen:", len(autobahn_as))

# 4) Koordinatensystem setzen:
#    BISStra nutzt UTM Zone 32N in ETRS89 -> EPSG:25832
#
#    Wir transformieren nach WGS84 (EPSG:4326), damit du Lat/Lon bekommst,
#    die du in anderen Tools, Karten, etc. nutzen kannst.
autobahn_as = autobahn_as.set_crs(epsg=25832)
autobahn_as_4326 = autobahn_as.to_crs(epsg=4326)

# 5) Längen- / Breitengrad als Spalten extrahieren
autobahn_as_4326["lon"] = autobahn_as_4326.geometry.x
autobahn_as_4326["lat"] = autobahn_as_4326.geometry.y

# 6) Relevante Spalten auswählen
#    Du kannst hier nach Bedarf noch mehr Attribute mitnehmen.
cols = [
    "NK_Kennung",
    "NK_Name",
    "NK_Knotenart",
    "NK_Knotenpunktfunktion",
    "NK_BABKnoten",
    "NK_GeoStrKls",
    "NK_x_utm32N",
    "NK_y_utm32N",
    "lon",
    "lat",
]

autobahn_as_out = autobahn_as_4326[cols].copy()

# 7) Als CSV speichern
out_csv = "autobahn_anschlussstellen_deutschland.csv"
autobahn_as_out.to_csv(out_csv, index=False, encoding="utf-8")

print(f"Fertig. Datei gespeichert als: {out_csv}")
print("Beispielzeilen:")
print(autobahn_as_out.head())
