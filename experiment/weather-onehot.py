import pandas as pd

# =========================
# CSV読み込み
# =========================
df = pd.read_csv("weather-clean.csv")

# =========================
# weatherコード対応表
# =========================
weather_dict = {
    1: "clear",
    2: "sunny",
    3: "partly_cloudy",
    4: "cloudy",
    5: "smoke",
    6: "sandstorm",
    7: "blizzard",
    8: "fog",
    9: "drizzle",
    10: "rain",
    11: "sleet",
    12: "snow",
    13: "hail_small",
    14: "hail",
    15: "thunder",
    16: "showers",
    17: "freezing_rain",
    18: "freezing_drizzle",
    19: "snow_showers",
    22: "snow_fog",
    23: "ice_rain",
    24: "ice_particles",
    28: "mist",
    101: "precipitation"
}

# =========================
# weather を名前へ変換
# =========================
df["weather_name"] = df["weather"].map(weather_dict)

# =========================
# ワンホットエンコーディング
# =========================
weather_onehot = pd.get_dummies(
    df["weather_name"],
    prefix="weather"
)

# True/False → 1/0 に変換
weather_onehot = weather_onehot.astype(int)

# =========================
# 元データへ結合
# =========================
df = pd.concat(
    [df, weather_onehot],
    axis=1
)

# 元の weather 列を消す場合
df = df.drop(
    columns=["weather", "weather_name"]
)

# =========================
# 表示
# =========================
print(df.to_string())

# =========================
# 保存
# =========================
df.to_csv(
    "weather-onehot.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nweather-onehot.csv saved.")