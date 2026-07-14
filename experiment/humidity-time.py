import pandas as pd

# CSV読み込み
df = pd.read_csv("merged-data.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# 学習データ
df = df[
    (df["datetime"] >= "2022-01-01") &
    (df["datetime"] < "2025-01-01")
]

df["hour"] = df["datetime"].dt.hour

# 欠損値削除
df = df.dropna(subset=["hour", "humidity"])

# 湿度を10%刻みに分類
bins = list(range(0, 110, 10))
labels = [
    "0-9","10-19","20-29","30-39","40-49",
    "50-59","60-69","70-79","80-89","90-100"
]

df["humidity_range"] = pd.cut(
    df["humidity"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# 件数表
table = pd.crosstab(
    df["hour"],
    df["humidity_range"]
)

# 時間帯ごとの割合(%)
table_percent = table.div(table.sum(axis=1), axis=0) * 100

print(table_percent.round(1))