import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# =========================
# CSV読み込み
# =========================
df = pd.read_csv("merged_data.csv")

# datetimeを日時型へ変換
df["datetime"] = pd.to_datetime(df["datetime"])

# =========================
# 学習データ（2022～2024年）
# =========================
train_df = df[
    (df["datetime"] >= "2022-01-01") &
    (df["datetime"] < "2025-01-01")
].copy()

# =========================
# 欠損値削除
# =========================
train_df = train_df.dropna(
    subset=[
        "temperature",
        "humidity",
        "solar",
        "wind_speed",
        "power_demand"
    ]
)

# =========================
# 使用特徴量
# =========================
features = [
    "temperature",
    "humidity",
    "solar",
    "wind_speed"
]

X = train_df[features]

# =========================
# 標準化
# =========================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# GMMクラスタリング
# =========================
gmm = GaussianMixture(
    n_components=4,
    random_state=0
)

train_df["cluster"] = gmm.fit_predict(X_scaled)

# =========================
# クラスタごとの統計量
# =========================

stats = train_df.groupby("cluster")[
    [
        "temperature",
        "humidity",
        "solar",
        "wind_speed",
        "power_demand"
    ]
].agg(["mean", "median", "max", "min"])

print("=== Cluster Statistics ===")
print(stats)

# =========================
# クラスタごとの代表値表示
# =========================

for c in sorted(train_df["cluster"].unique()):

    print("\n====================================")
    print(f"Cluster {c}")
    print("====================================")

    cluster = train_df[train_df["cluster"] == c]

    for feature in features:

        mean = cluster[feature].mean()
        std = cluster[feature].std()
        q1 = cluster[feature].quantile(0.25)
        median = cluster[feature].median()
        q3 = cluster[feature].quantile(0.75)

        print(f"{feature}")
        print(f"  平均値   : {mean:.2f}")
        print(f"  標準偏差 : {std:.2f}")
        print(f"  第1四分位: {q1:.2f}")
        print(f"  中央値   : {median:.2f}")
        print(f"  第3四分位: {q3:.2f}")
        print()

# =========================
# クラスタごとの全データ表示
# =========================

for c in sorted(train_df["cluster"].unique()):

    print("\n" + "=" * 60)
    print(f"Cluster {c} の全データ")
    print("=" * 60)

    print(
        train_df[
            train_df["cluster"] == c
        ][
            [
                "datetime",
                "temperature",
                "humidity",
                "solar",
                "wind_speed",
                "power_demand"
            ]
        ].sample(n=20)
    )
# =========================
# クラスタ数
# =========================
print("\n=== Cluster Count ===")
print(train_df["cluster"].value_counts())

# =========================
# 各データの所属確率
# =========================
prob = gmm.predict_proba(X_scaled)

prob_df = pd.DataFrame(
    prob,
    columns=[
        "Cluster0",
        "Cluster1",
        "Cluster2",
        "Cluster3"
    ]
)

print("\n=== Membership Probability ===")
print(prob_df.head(10))

# =========================
# PCAで2次元化
# =========================
pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(X_scaled)

train_df["PCA1"] = X_pca[:, 0]
train_df["PCA2"] = X_pca[:, 1]

print("\n=== Explained Variance Ratio ===")
print(pca.explained_variance_ratio_)
print("Cumulative =", pca.explained_variance_ratio_.sum())

# =========================
# PCA散布図
# =========================
plt.figure(figsize=(8,6))

for c in sorted(train_df["cluster"].unique()):

    data = train_df[
        train_df["cluster"] == c
    ]

    plt.scatter(
        data["PCA1"],
        data["PCA2"],
        alpha=0.5,
        label=f"Cluster {c}"
    )

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("GMM Clustering (Training Data)")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# クラスタごとのPCA散布図（軸固定）
# =========================

xmin = train_df["PCA1"].min()
xmax = train_df["PCA1"].max()
ymin = train_df["PCA2"].min()
ymax = train_df["PCA2"].max()

for c in sorted(train_df["cluster"].unique()):

    data = train_df[
        train_df["cluster"] == c
    ]

    plt.figure(figsize=(8,6))

    plt.scatter(
        data["PCA1"],
        data["PCA2"],
        alpha=0.6
    )

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.xlabel("PCA1")
    plt.ylabel("PCA2")

    plt.title(f"Cluster {c} (PCA)")

    plt.grid(True)

    plt.show()
# =========================
# クラスタごとの特徴表示
# =========================
for c in sorted(train_df["cluster"].unique()):

    print("\n===========================")
    print(f"Cluster {c}")
    print("===========================")

    print(
        train_df[
            train_df["cluster"] == c
        ][
            [
                "temperature",
                "humidity",
                "solar",
                "wind_speed",
                "power_demand"
            ]
        ].mean()
    )

    plt.figure(figsize=(7,4))

    plt.bar(
        features,
        train_df[
            train_df["cluster"] == c
        ][features].mean()
    )

    plt.title(f"Cluster {c}")

    plt.ylabel("Average")

    plt.grid(True)

    plt.show()

print("\n=== PCA Components ===")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PCA1", "PCA2"],
    index=features
)

print(loadings)