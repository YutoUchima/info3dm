import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# CSV読み込み
# =========================
df = pd.read_csv("merged-data.csv")

# datetime を日時型へ変換
df["datetime"] = pd.to_datetime(df["datetime"])

# =========================
# 学習データ
# 2022〜2024
# =========================
train_df = df[
    (df["datetime"] >= "2022-01-01") &
    (df["datetime"] < "2025-01-01")
]

# =========================
# テストデータ
# 2025〜2026/5/31
# =========================
test_df = df[
    (df["datetime"] >= "2025-01-01") &
    (df["datetime"] <= "2026-05-31 23:59:59")
]




from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
# =========================
# GMMクラスタリング
# =========================

cluster_features = [
    "temperature",
    "humidity",
    "solar",
    "wind_speed"
]

# 欠損値削除
train_df = train_df.dropna(subset=cluster_features)
test_df = test_df.dropna(subset=cluster_features)

# 標準化
scaler = StandardScaler()

X_cluster_train = scaler.fit_transform(
    train_df[cluster_features]
)

X_cluster_test = scaler.transform(
    test_df[cluster_features]
)

# GMM学習
gmm = GaussianMixture(
    n_components=4,
    random_state=0
)

train_df["cluster"] = gmm.fit_predict(
    X_cluster_train
)

# テストデータには学習済みGMMでクラスタ付与
test_df["cluster"] = gmm.predict(
    X_cluster_test
)

# =========================
# 特徴量
# =========================
features = [
    "temperature",
    "humidity",
    "solar",
    "wind_speed",
    "cluster",
    "is_holiday",
    "is_weekday",
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
    "hour_23",
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
    "month_12",
    "weather_clear",
    "weather_cloudy",
    "weather_drizzle",
    "weather_fog",
    "weather_partly_cloudy",
    "weather_sunny",
    "weather_thunder"
]

# =========================
# 学習データ
# =========================
X_train = train_df[features]
y_train = train_df["power_demand"]

# =========================
# テストデータ
# =========================
X_test = test_df[features]
y_test = test_df["power_demand"]

# 日時保存（表示用）
date_test = test_df["datetime"]

# =========================
# Random Forest モデル
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
print("=== Random Forest Evaluation ===")

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R² =", r2)
print("MSE =", mse)

# RMSE
rmse = mse ** 0.5
print("RMSE =", rmse)

# =========================
# 特徴量重要度
# =========================
print("\n=== Feature Importance ===")

for feature, importance in zip(
    features,
    model.feature_importances_
):
    print(f"{feature}: {importance:.4f}")

# =========================
# 予測結果表示
# =========================
result = pd.DataFrame({
    "Date": date_test.values,
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\n=== Prediction Results ===")
print(result.head(20))


