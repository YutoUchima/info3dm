import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# =========================
# CSV読み込み
# =========================
df = pd.read_csv("merged-data.csv")

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

]


# hour追加
for i in range(24):
    features.append(f"hour_{i}")


# month追加
for i in range(1,13):
    features.append(f"month_{i}")


# weather追加
weather_features = [
    "weather_clear",
    "weather_cloudy",
    "weather_drizzle",
    "weather_fog",
    "weather_partly_cloudy",
    "weather_sunny",
    "weather_thunder"
]

features += weather_features



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
# 予測条件入力
# =========================

print("\n=== 予測条件入力 ===")


temperature = float(
    input("気温(℃)：")
)

humidity = float(
    input("湿度(%)：")
)

solar = float(
    input("日照量：")
)

wind_speed = float(
    input("風速(m/s)：")
)


holiday = int(
    input("休日なら1 平日なら0：")
)


weekday = 1 - holiday


hour = int(
    input("予測する時間(0〜23)：")
)


month = int(
    input("予測する月(1〜12)：")
)


print("""
天気を入力してください
1:clear
2:cloudy
3:drizzle
4:fog
5:partly_cloudy
6:sunny
7:thunder
""")


weather = int(
    input("天気番号：")
)



# =========================
# 入力データ作成
# =========================

future = {

    "temperature": temperature,
    "humidity": humidity,
    "solar": solar,
    "wind_speed": wind_speed,

    "is_holiday": holiday,
    "is_weekday": weekday

}



# hourワンホット
for i in range(24):

    if i == hour:
        future[f"hour_{i}"] = 1
    else:
        future[f"hour_{i}"] = 0



# monthワンホット
for i in range(1,13):

    if i == month:
        future[f"month_{i}"] = 1
    else:
        future[f"month_{i}"] = 0



# weatherワンホット
for i, name in enumerate(weather_features, start=1):

    if i == weather:
        future[name] = 1
    else:
        future[name] = 0



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



print("\n====================")
print("予測電気消費量")
print(round(prediction[0],2),"kW")
print("====================")