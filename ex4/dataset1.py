# # フォント設定
# import matplotlib
# import matplotlib.font_manager as font_manager
# font_path = '/Library/Fonts/Arial Unicode.ttf'
# font_prop = font_manager.FontProperties(fname = font_path)
# matplotlib.rcParams['font.family'] = font_prop.get_name()

# # ex1.1
# import numpy as np
# import matplotlib.pyplot as plt
# def true_function(x):
#     return np.sin(np.pi * x * 0.8) * 10
    
# x = 0
# y = true_function(x)
# print(x, y)
# x = np.linspace(-1, 1, 100)
# y = true_function(x)
# plt.plot(x, y, label='y = 10 sin(0.8πx)')
# plt.legend()
# plt.savefig("ex1.1.png")
# plt.show()


# # ex1.2
# import pandas as pd
# np.random.seed(0)
# n = 20
# x_random = np.random.uniform(-1, 1, n)
# y_random = true_function(x_random)
# df = pd.DataFrame({
#     "観測点": x_random,
#     "真値": y_random
# })
# print(df)  

# plt.figure(figsize=(8, 5))  
# plt.plot(x, y, label='y = 10 sin(0.8πx)')
# plt.scatter(x_random, y_random, color="red", label="観測点")
# plt.legend()
# plt.savefig("ex1.2.png")
# plt.show()


# # ex1.3
# noise = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=n) / 2
# observed_value = y_random + noise
# df["観測値"] = observed_value
# print(df)
# plt.figure(figsize=(8, 5))
# plt.plot(x, y, label='y = 10 sin(0.8πx)')
# plt.scatter(x_random, y_random, color="red", label="真値")
# plt.scatter(x_random, observed_value, color="green", label="観測値")
# plt.legend()
# plt.savefig("ex1.3.png")
# plt.show()


# # ex1.4
# df.to_csv("ex1.4.tsv", sep="\t", index=False)


# # ex1.5
# import pandas as pd
# df = pd.read_csv("ex1.4.tsv", sep="\t")
# print(df)


# ex1.6
# dataset1.py

# フォント設定
import matplotlib
import matplotlib.font_manager as font_manager

font_path = '/Library/Fonts/Arial Unicode.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

# ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ex1.1
def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10


def ex1_1():
    x = np.linspace(-1, 1, 100)
    y = true_function(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='y = 10 sin(0.8πx)')
    plt.legend()
    plt.savefig("ex1.1.png")
    plt.show()
    
    return x, y


# ex1.2
def ex1_2(seed=0, n=20):
    np.random.seed(seed)

    x_random = np.random.uniform(-1, 1, n)
    y_random = true_function(x_random)

    df = pd.DataFrame({
        "観測点": x_random,
        "真値": y_random
    })
    x = np.linspace(-1, 1, 100)
    y = true_function(x)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='y = 10 sin(0.8πx)')
    plt.scatter(x_random, y_random, color="red", label="観測点")
    plt.legend()
    plt.savefig("ex1.2.png")
    plt.show()

    return df


# ex1.3
def ex1_3(df, seed=0):
    np.random.seed(seed)

    noise = np.random.normal(
        loc=0.0,
        scale=np.sqrt(2.0),
        size=len(df)
    ) / 2

    observed_value = df["真値"] + noise

    df["観測値"] = observed_value

    x = np.linspace(-1, 1, 100)
    y = true_function(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='y = 10 sin(0.8πx)')

    plt.scatter(
        df["観測点"],
        df["真値"],
        color="red",
        label="真値"
    )

    plt.scatter(
        df["観測点"],
        df["観測値"],
        color="green",
        label="観測値"
    )

    plt.legend()
    plt.savefig("ex1.3.png")
    plt.show()

    return df


# ex1.4
def ex1_4(df, filename="ex1.4.tsv"):
    df.to_csv(filename, sep="\t", index=False)


# ex1.5
def ex1_5(filename="ex1.4.tsv"):
    df = pd.read_csv(filename, sep="\t")
    return df