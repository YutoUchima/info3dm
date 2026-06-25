import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# =========================
# CSV読み込み
# =========================
df = pd.read_csv("merged-data.csv")

# datetime変換
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
# 特徴量
# =========================
features = [

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


# =========================
# 学習
# =========================

X_train = train_df[features]

y_train = train_df["power_demand"]


model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)


model.fit(
    X_train,
    y_train
)


print("学習完了")


# =========================
# 未来の予測データ入力
# =========================

future = {

    "temperature": 32.3,
    "humidity": 82,
    "solar": 3.07,
    "wind_speed": 4.7,

    "is_holiday": 0,
    "is_weekday": 1,


    # 14時の場合
    "hour_0":0,
    "hour_1":0,
    "hour_2":0,
    "hour_3":0,
    "hour_4":0,
    "hour_5":0,
    "hour_6":0,
    "hour_7":0,
    "hour_8":0,
    "hour_9":0,
    "hour_10":0,
    "hour_11":0,
    "hour_12":0,
    "hour_13":1,
    "hour_14":0,
    "hour_15":0,
    "hour_16":0,
    "hour_17":0,
    "hour_18":0,
    "hour_19":0,
    "hour_20":0,
    "hour_21":0,
    "hour_22":0,
    "hour_23":0,


    # 6月の場合
    "month_1":0,
    "month_2":0,
    "month_3":0,
    "month_4":0,
    "month_5":0,
    "month_6":1,
    "month_7":0,
    "month_8":0,
    "month_9":0,
    "month_10":0,
    "month_11":0,
    "month_12":0,


    # 天気（晴れ）
    "weather_clear":0,
    "weather_cloudy":1,
    "weather_drizzle":0,
    "weather_fog":0,
    "weather_partly_cloudy":0,
    "weather_sunny":0,
    "weather_thunder":0

}



# DataFrame化

future_df = pd.DataFrame(
    [future]
)


# =========================
# 予測
# =========================

prediction = model.predict(
    future_df[features]
)


print("====================")
print("予測電気消費量")
print(prediction[0], "kW")
print("====================")