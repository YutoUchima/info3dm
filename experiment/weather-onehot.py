import pandas as pd
import jpholiday

# =========================
# CSV読み込み
# =========================
df = pd.read_csv("weather-clean.csv")

# datetime を日時型へ変換
df["datetime"] = pd.to_datetime(df["datetime"])

# =========================
# hour ワンホット
# =========================

# 時間取得
df["hour"] = df["datetime"].dt.hour

# ワンホット化
hour_onehot = pd.get_dummies(
    df["hour"],
    prefix="hour"
).astype(int)

# =========================
# month ワンホット
# =========================

# 月取得
df["month"] = df["datetime"].dt.month

# ワンホット化
month_onehot = pd.get_dummies(
    df["month"],
    prefix="month"
).astype(int)

# =========================
# 曜日取得
# =========================

# 月=0, 日=6
df["weekday"] = df["datetime"].dt.weekday

# 日付だけ取得
df["date_only"] = df["datetime"].dt.date

# =========================
# 土日判定
# =========================
is_weekend = df["weekday"] >= 5

# =========================
# 日本の祝日判定
# =========================
is_japanese_holiday = df["date_only"].apply(
    lambda x: jpholiday.is_holiday(x)
)

# =========================
# 独自休日
# =========================
custom_holidays = [
    "2022-06-23",
    "2023-06-23",
    "2024-06-23",
    "2025-06-23",
    "2026-06-23"
]

# datetime.date 型へ変換
custom_holidays = pd.to_datetime(
    custom_holidays
).date

# 独自休日判定
is_custom_holiday = df["date_only"].isin(
    custom_holidays
)

# =========================
# 休日フラグ
# =========================
df["is_holiday"] = (
    is_weekend |
    is_japanese_holiday |
    is_custom_holiday
).astype(int)

# =========================
# 平日フラグ
# =========================
df["is_weekday"] = (
    df["is_holiday"] == 0
).astype(int)

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
df["weather_name"] = df["weather"].map(
    weather_dict
)

# =========================
# weather ワンホット
# =========================
weather_onehot = pd.get_dummies(
    df["weather_name"],
    prefix="weather"
).astype(int)

# =========================
# 全特徴量結合
# =========================
df = pd.concat(
    [
        df,
        hour_onehot,
        month_onehot,
        weather_onehot
    ],
    axis=1
)

# =========================
# 不要列削除
# =========================
df = df.drop(
    columns=[
        "weather",
        "weather_name",
        "hour",
        "month",
        "weekday",
        "date_only"
    ]
)

# =========================
# 確認
# =========================
print(df.head(20).to_string())

# =========================
# 保存
# =========================
df.to_csv(
    "weather-onehot.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nweather-onehot.csv saved.")