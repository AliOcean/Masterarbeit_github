# ============================
# 1. Imports
# ============================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================
# 2. Daten laden
# ============================
df = pd.read_csv("train_dataset_final.csv")

# Timestamp als Datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Nach Zeit sortieren (wichtig für zeitbasierten Split!)
df = df.sort_values("timestamp")

# ============================
# 3. Features & Target
# ============================
target = "occupancy_percent"

features = [
    "rain",
    "temperature",
    "hour",
    "weekday",
    "capacity",
    "latitude",
    "longitude"
]

X = df[features]
y = df[target]

# ============================
# 4. Zeitbasierte Train/Test-Trennung (80/20)
# ============================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ============================
# 5. Random Forest trainieren
# ============================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ============================
# 6. Vorhersage
# ============================
y_pred = model.predict(X_test)

# ============================
# 7. Performance-Metriken
# ============================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))

# ============================
# 8. Feature Importance
# ============================
importances = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(importances)
