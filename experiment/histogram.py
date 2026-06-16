import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CSV読み込み
# =========================
df = pd.read_csv("merged-data.csv")

# =========================
# 通常特徴量
# =========================
normal_features = [
    "temperature",
    "humidity",
    "solar",
    "wind_speed",
    "is_holiday",
    "is_weekday",
    "weather_clear",
    "weather_cloudy",
    "weather_drizzle",
    "weather_fog",
    "weather_partly_cloudy",
    "weather_sunny"
]

# =========================
# 通常特徴量の散布図
# =========================
for feature in normal_features:

    # 列が存在しない場合スキップ
    if feature not in df.columns:
        continue

    # 相関係数
    corr = df[feature].corr(
        df["power_demand"]
    )

    # グラフ
    plt.figure(figsize=(8, 6))

    plt.scatter(
        df[feature],
        df["power_demand"],
        alpha=0.5
    )

    plt.title(
        f"{feature} vs Power Demand\nCorrelation: {corr:.3f}"
    )

    plt.xlabel(feature)
    plt.ylabel("Power Demand")

    plt.grid(True)
    plt.tight_layout()

# =========================
# hour を1つのグラフに表示
# =========================
plt.figure(figsize=(12, 8))

for i in range(24):

    col = f"hour_{i}"

    # 列が存在する場合のみ
    if col not in df.columns:
        continue

    # hour_i = 1 のデータだけ抽出
    hour_data = df[df[col] == 1]

    plt.scatter(
        [i] * len(hour_data),
        hour_data["power_demand"],
        alpha=0.5,
        label=f"{i}:00"
    )

plt.title("Hour vs Power Demand")

plt.xlabel("Hour")
plt.ylabel("Power Demand")

plt.xticks(range(24))

plt.grid(True)
plt.tight_layout()

# =========================
# month を1つのグラフに表示
# =========================
plt.figure(figsize=(12, 8))

for i in range(1, 13):

    col = f"month_{i}"

    if col not in df.columns:
        continue

    month_data = df[df[col] == 1]

    plt.scatter(
        [i] * len(month_data),
        month_data["power_demand"],
        alpha=0.5,
        label=f"Month {i}"
    )

plt.title("Month vs Power Demand")

plt.xlabel("Month")
plt.ylabel("Power Demand")

plt.xticks(range(1, 13))

plt.grid(True)
plt.tight_layout()

# =========================
# 表示
# =========================
plt.show()

# =========================
# 相関係数一覧
# =========================
print("=== Correlation Coefficients ===")

for feature in normal_features:

    if feature not in df.columns:
        continue

    corr = df[feature].corr(
        df["power_demand"]
    )

    print(f"{feature}: {corr:.3f}")