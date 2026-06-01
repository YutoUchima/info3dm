import pandas as pd

# 天気データ読み込み
weather = pd.read_csv(
    "weather-clean.csv",
    parse_dates=["datetime"]
)

# 電力データ読み込み
power = pd.read_csv(
    "denki-clean.csv",
    parse_dates=["datetime"]
)

# datetime をキーに結合
merged = pd.merge(
    weather,
    power,
    on="datetime",
    how="inner"
)

# 確認
print("\n===== MERGED DATA =====")
print(merged.to_string())

# 保存
merged.to_csv("merged-data.csv", index=False)

print("\nmerged-data.csv saved.")