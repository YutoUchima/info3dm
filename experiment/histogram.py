import pandas as pd
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("merged-data.csv")

# 散布図を描く特徴量一覧
features = [
    "temperature",
    "humidity",
    "solar",
    "wind_speed"
]

# 特徴量ごとに散布図 + 相関係数
for feature in features:

    # 相関係数計算
    corr = df[feature].corr(df["power_demand"])

    # グラフ作成
    plt.figure(figsize=(8, 6))

    plt.scatter(
        df[feature],
        df["power_demand"]
    )

    # タイトルに相関係数表示
    plt.title(
        f"{feature} vs Power Demand\nCorrelation: {corr:.3f}"
    )

    # 軸ラベル
    plt.xlabel(feature)
    plt.ylabel("Power Demand")

    # グリッド
    plt.grid(True)

    # レイアウト調整
    plt.tight_layout()

# 表示
plt.show()

# 相関係数一覧表示
print("=== Correlation Coefficients ===")

for feature in features:
    corr = df[feature].corr(df["power_demand"])
    print(f"{feature}: {corr:.3f}")