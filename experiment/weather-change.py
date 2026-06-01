import pandas as pd
import glob

# =========================
# 全CSV取得
# =========================
files = glob.glob("weather-data/*.csv")

weather_list = []

# =========================
# 各CSV読み込み
# =========================
for file in files:

    weather = pd.read_csv(
        file,
        encoding="cp932",
        skiprows=4
    )

    # =========================
    # 必要列抽出
    # =========================
    weather = weather.iloc[:, [
        0,   # datetime
        1,   # temperature
        4,   # rainfall
        16,  # solar
        22,  # wind_speed
        27,  # weather
        30   # humidity
    ]]

    # =========================
    # 列名変更
    # =========================
    weather.columns = [
        "datetime",
        "temperature",
        "rainfall",
        "solar",
        "wind_speed",
        "weather",
        "humidity"
    ]

    # =========================
    # datetime変換
    # =========================
    weather["datetime"] = pd.to_datetime(
        weather["datetime"],
        errors="coerce"
    )

    # datetime欠損削除
    weather = weather.dropna(
        subset=["datetime"]
    )

    # =========================
    # 数値型変換
    # =========================
    numeric_columns = [
        "temperature",
        "rainfall",
        "solar",
        "wind_speed",
        "weather",
        "humidity"
    ]

    for col in numeric_columns:
        weather[col] = pd.to_numeric(
            weather[col],
            errors="coerce"
        )

    # リスト追加
    weather_list.append(weather)

# =========================
# 全データ結合
# =========================
all_weather = pd.concat(
    weather_list,
    ignore_index=True
)

# 時系列順ソート
all_weather = all_weather.sort_values(
    "datetime"
)

# =========================
# 表示
# =========================
print("\n===== WEATHER DATA =====")
print(all_weather.to_string())

# =========================
# 保存
# =========================
all_weather.to_csv(
    "weather-clean.csv",
    index=False,
    encoding="utf-8"
)

print("\nweather-clean.csv saved.")