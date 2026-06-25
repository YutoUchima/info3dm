import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

from sklearn.inspection import PartialDependenceDisplay


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
    "is_weekday",

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
    "hour_23",

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
    "month_12",

    # weather
    "weather_clear",
    "weather_cloudy",
    "weather_drizzle",
    "weather_fog",
    "weather_partly_cloudy",
    "weather_sunny"
]


# 存在する特徴量だけ使用
features = [
    col for col in features
    if col in df.columns
]


print("=== USED FEATURES ===")
print(features)



# =========================
# 学習データ
# 2022〜2024
# =========================

df["datetime"] = pd.to_datetime(
    df["datetime"]
)


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



# =========================
# X,y
# =========================

X_train = train_df[features]

y_train = train_df["power_demand"]


X_test = test_df[features]

y_test = test_df["power_demand"]



# =========================
# Random Forest
# =========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)



# 学習
model.fit(
    X_train,
    y_train
)



# =========================
# 予測
# =========================

y_pred = model.predict(
    X_test
)



# =========================
# 評価
# =========================

print("\n=== Evaluation ===")

r2 = r2_score(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

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


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


print("\n=== Feature Importance ===")

print(importance_df)



# =========================
# 重要度グラフ
# =========================

plt.figure(figsize=(12,8))


plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)


plt.title(
    "Random Forest Feature Importance"
)


plt.gca().invert_yaxis()


plt.tight_layout()



# =========================
# 実測値 vs 予測値
# =========================

plt.figure(figsize=(8,8))


plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)


plt.xlabel(
    "Actual"
)

plt.ylabel(
    "Predicted"
)


plt.title(
    "Actual vs Predicted"
)


plt.grid(True)


plt.tight_layout()



# =========================
# Partial Dependence Plot
# 非線形関係
# =========================

pdp_features = [
    "temperature",
    "humidity",
    "solar",
    "wind_speed",
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


for feature in pdp_features:

    if feature in features:

        fig, ax = plt.subplots(
            figsize=(8,6)
        )


        PartialDependenceDisplay.from_estimator(
            model,
            X_test,
            [feature],
            ax=ax
        )


        plt.title(
            f"Partial Dependence Plot : {feature}"
        )


        plt.tight_layout()



# =========================
# 表示
# =========================

plt.show()