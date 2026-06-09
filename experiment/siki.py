
import pandas as pd
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("merged-data.csv")

# 散布図を描く特徴量一覧
features = [
    "temperature",
    "humidity",
    "solar",
    "wind_speed",
    "weather_clear",
    "weather_cloudy",
    "weather_drizzle",
    "weather_fog",
    "weather_partly_cloudy",
    "weather_sunny",
    "weather_thunder"
]

# =====================
# 線形回帰
# =====================

from sklearn.linear_model import LinearRegression

# 特徴量と目的変数
X = df[features]
y = df["power_demand"]

# モデル作成
model = LinearRegression()

# 学習（フィッティング）
model.fit(X, y)

# 回帰係数表示
print("\n=== Linear Regression Coefficients ===")

for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")

# 切片表示
print(f"Intercept: {model.intercept_:.4f}")


equation = "power_demand = "

for i, (feature, coef) in enumerate(zip(features, model.coef_)):
    if i > 0:
        equation += " + "
    equation += f"({coef:.4f} × {feature})"

equation += f" + ({model.intercept_:.4f})"

print("\n=== Regression Equation ===")
print(equation)