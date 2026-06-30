import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =========================
# CSV読み込み
# =========================

df = pd.read_csv(
    "merged-data.csv"
)


# =========================
# 欠損削除
# =========================

df = df.dropna(
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


X = df[features]



# =========================
# 標準化
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)



# =========================
# K-means
# =========================

kmeans = KMeans(
    n_clusters=4,
    random_state=0
)


df["cluster"] = kmeans.fit_predict(
    X_scaled
)



# =========================
# クラスタ平均
# =========================

cluster_result = df.groupby(
    "cluster"
)[
    [
        "temperature",
        "humidity",
        "solar",
        "wind_speed",
        "power_demand"
    ]
].mean()


print("=== Cluster Mean ===")
print(cluster_result)



# =========================
# クラスタごとのデータ数
# =========================

print("\n=== Cluster Count ===")

print(
    df["cluster"].value_counts()
)



# =========================
# クラスタごとに表示
# =========================

for c in sorted(df["cluster"].unique()):

    cluster_data = df[
        df["cluster"] == c
    ]


    print("\n===================")
    print(f"Cluster {c}")
    print("===================")

    print(
        cluster_data[
            [
                "temperature",
                "humidity",
                "solar",
                "wind_speed",
                "power_demand"
            ]
        ].mean()
    )


    # -------------------------
    # 気象4要素グラフ
    # -------------------------

    plt.figure(
        figsize=(8,4)
    )


    plt.bar(
        features,
        cluster_data[features].mean()
    )


    plt.title(
        f"Weather Features Cluster {c}"
    )

    plt.ylabel(
        "Average Value"
    )


    plt.grid(True)

    plt.show()



    # =========================
# クラスタごとの分布表示
# =========================

clusters = sorted(
    df["cluster"].unique()
)


for c in clusters:

    data = df[
        df["cluster"] == c
    ]


    plt.figure(
        figsize=(8,6)
    )


    plt.scatter(
        data["temperature"],
        data["solar"],
        alpha=0.5
    )


    plt.xlabel(
        "Temperature"
    )

    plt.ylabel(
        "Solar"
    )


    plt.title(
        f"Cluster {c}: Temperature vs Solar"
    )


    plt.grid(True)


    plt.show()


# =========================
# PCAによる次元削減
# =========================

from sklearn.decomposition import PCA


# 2次元へ削減
pca = PCA(
    n_components=2
)


X_pca = pca.fit_transform(
    X_scaled
)



# =========================
# PCA結果をDataFrameへ追加
# =========================

df["PCA1"] = X_pca[:,0]
df["PCA2"] = X_pca[:,1]



# =========================
# 寄与率表示
# =========================

print("\n=== PCA Explained Variance Ratio ===")

print(
    pca.explained_variance_ratio_
)


print(
    "累積寄与率:",
    sum(pca.explained_variance_ratio_)
)



# =========================
# PCA散布図
# =========================

plt.figure(
    figsize=(8,6)
)


for c in sorted(df["cluster"].unique()):

    data = df[
        df["cluster"] == c
    ]


    plt.scatter(
        data["PCA1"],
        data["PCA2"],
        alpha=0.6,
        label=f"Cluster {c}"
    )



plt.xlabel(
    "PCA Component 1"
)


plt.ylabel(
    "PCA Component 2"
)


plt.title(
    "K-means Clusters Visualized by PCA"
)


plt.legend()


plt.grid(True)


plt.show()