import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# CSV読み込み
df = pd.read_csv("merged-data.csv")

# 特徴量
features = [
    "temperature",
    "humidity",
    "solar",
    "wind_speed",
    "weather_clear",
    "weather_cloudy",
    "weather_drizzle",
    "weather_fog",
    "weather_partly_cloudy",
    "weather_sunny",
    "weather_thunder"
    
]

# 入力データ(X)と正解データ(y)
X = df[features]
y = df["power_demand"]
dates = df["datetime"]

X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
    X,
    y,
    dates,
    test_size=0.2,
    random_state=0
)

# Random Forest モデル作成
model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)

# 学習
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 評価
print("=== Random Forest Evaluation ===")
print("R² =", r2_score(y_test, y_pred))
print("MSE =", mean_squared_error(y_test, y_pred))

# 特徴量重要度
print("\n=== Feature Importance ===")

for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# 予測結果表示
result = pd.DataFrame({
    "Date": date_test.values,
    "Actual": y_test.values,
    "Predicted": y_pred
})



print("\n=== Prediction Results ===")
print(result.head(20))