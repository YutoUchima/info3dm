import pandas as pd
import matplotlib.pyplot as plt

# =====================
# CSV読み込み
# =====================
df = pd.read_csv("merged-data.csv")

# =====================
# weather 系 NaN確認用
# =====================
print("=== NaN BEFORE ===")
print(df.isnull().sum())

# =====================
# 使いたい特徴量候補
# =====================
candidate_features = [

    # 気象
    "temperature",
    "humidity",
    "solar",
    "wind_speed",

    # 休日
    "is_holiday",

    # hour
    "hour_0",
    "hour_1",
    "hour_2",
    "hour_3",
    "hour_4",
    "hour_5",
    "hour_6",
    "hour_7",
    "hour_8",
    "hour_9",
    "hour_10",
    "hour_11",
    "hour_12",
    "hour_13",
    "hour_14",
    "hour_15",
    "hour_16",
    "hour_17",
    "hour_18",
    "hour_19",
    "hour_20",
    "hour_21",
    "hour_22",
    # hour_23 は削除（多重共線性対策）

    # month
    "month_1",
    "month_2",
    "month_3",
    "month_4",
    "month_5",
    "month_6",
    "month_7",
    "month_8",
    "month_9",
    "month_10",
    "month_11",
    # month_12 は削除（多重共線性対策）

    # weather
    "weather_clear",
    "weather_cloudy",
    "weather_drizzle",
    "weather_fog",
    "weather_partly_cloudy",
    "weather_sunny"
    # weather_thunder など存在しない可能性があるので除外
]

# =====================
# 実際に存在する列だけ使う
# =====================
features = [
    col for col in candidate_features
    if col in df.columns
]

print("\n=== USED FEATURES ===")
print(features)

# =====================
# 欠損値確認
# =====================
print("\n=== NaN in Features ===")
print(df[features].isnull().sum())

# =====================
# 欠損値行削除
# =====================
df = df.dropna(
    subset=features + ["power_demand"]
)

print("\n=== DATA SIZE AFTER DROPNA ===")
print(df.shape)

# =====================
# 線形回帰
# =====================
from sklearn.linear_model import LinearRegression

# 特徴量と目的変数
X = df[features]
y = df["power_demand"]

# モデル作成
model = LinearRegression()

# 学習
model.fit(X, y)

# =====================
# 回帰係数表示
# =====================
print("\n=== Linear Regression Coefficients ===")

for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")

# 切片
print(f"\nIntercept: {model.intercept_:.4f}")

# =====================
# 回帰式表示
# =====================
equation = "power_demand = "

for i, (feature, coef) in enumerate(
    zip(features, model.coef_)
):
    if i > 0:
        equation += " + "

    equation += f"({coef:.4f} × {feature})"

equation += f" + ({model.intercept_:.4f})"

print("\n=== Regression Equation ===")
print(equation)