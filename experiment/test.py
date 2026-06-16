import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

# =========================
# CSV読み込み
# =========================
df = pd.read_csv("merged-data.csv")

# =========================
# 欠損値削除
# =========================
df = df.dropna()

# =========================
# 特徴量
# =========================
features = [

    # 気象
    "temperature",
    "humidity",
    "solar",
    "wind_speed",

    # 休日
    "is_holiday",

    # hour
    "hour_0",
    "hour_1",
    "hour_2",
    "hour_3",
    "hour_4",
    "hour_5",
    "hour_6",
    "hour_7",
    "hour_8",
    "hour_9",
    "hour_10",
    "hour_11",
    "hour_12",
    "hour_13",
    "hour_14",
    "hour_15",
    "hour_16",
    "hour_17",
    "hour_18",
    "hour_19",
    "hour_20",
    "hour_21",
    "hour_22",

    # month
    "month_1",
    "month_2",
    "month_3",
    "month_4",
    "month_5",
    "month_6",
    "month_7",
    "month_8",
    "month_9",
    "month_10",
    "month_11",

    # weather
    "weather_clear",
    "weather_cloudy",
    "weather_drizzle",
    "weather_fog",
    "weather_partly_cloudy",
    "weather_sunny"
]

# 実際に存在する列だけ使う
features = [
    col for col in features
    if col in df.columns
]

# =========================
# 学習データ
# =========================
train_df = df[
    (pd.to_datetime(df["datetime"]) >= "2022-01-01") &
    (pd.to_datetime(df["datetime"]) < "2025-01-01")
]

# =========================
# テストデータ
# =========================
test_df = df[
    (pd.to_datetime(df["datetime"]) >= "2025-01-01") &
    (pd.to_datetime(df["datetime"]) <= "2026-05-31")
]

# =========================
# X, y
# =========================
X_train = train_df[features]
y_train = train_df["power_demand"]

X_test = test_df[features]
y_test = test_df["power_demand"]

# =========================
# モデル作成
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)

# =========================
# 学習
# =========================
model.fit(X_train, y_train)

# =========================
# 予測
# =========================
y_pred = model.predict(X_test)

# =========================
# 評価
# =========================
print("=== Evaluation ===")

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"R²   : {r2:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")

# =========================
# 特徴量重要度
# =========================
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

# 大きい順
importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\n=== Feature Importance ===")
print(importance_df)

# =========================
# 特徴量重要度グラフ
# =========================
plt.figure(figsize=(12, 8))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

# =========================
# 実測値 vs 予測値
# =========================
plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.title(
    "Actual vs Predicted"
)

plt.grid(True)

plt.tight_layout()

# =========================
# 表示
# =========================


from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    ["temperature"]
)

plt.show()
