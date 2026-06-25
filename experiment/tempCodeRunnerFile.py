import pandas as pd
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("denki-clean.csv")

# datetime を日時型に変換
df["datetime"] = pd.to_datetime(df["datetime"])

# 日付と時間を分離
df["DATE"] = df["datetime"].dt.date
df["TIME"] = df["datetime"].dt.strftime("%H:%M")

# 日付一覧
dates = df["DATE"].unique()

# グラフサイズ
plt.figure(figsize=(14, 8))

# 日付ごとに描画
for date in dates:

    # その日のデータ
    day_data = df[df["DATE"] == date]

    # x軸
    x = range(len(day_data))

    # グラフ
    plt.plot(
        x,
        day_data["power_demand"],
        marker='o',
        linestyle='-',
        label=str(date)
    )

# x軸ラベル
plt.xticks(x, day_data["TIME"], rotation=45)

# タイトル
plt.title("Daily Power Demand Comparison")

# 軸ラベル
plt.xlabel("Time")
plt.ylabel("Power Demand")

# 凡例
plt.legend()

# グリッド
plt.grid(True)

# レイアウト調整
plt.tight_layout()

# 表示
plt.show()