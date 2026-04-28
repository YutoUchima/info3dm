# フォント設定
import matplotlib
import matplotlib.font_manager as font_manager
font_path = '/Library/Fonts/Arial Unicode.ttf'
font_prop = font_manager.FontProperties(fname = font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

np.random.seed(0)
n = 20
x_random = np.random.uniform(-1, 1, n)
y_random = true_function(x_random)

df = pd.DataFrame({
    "観測点": x_random,
    "真値": y_random
})

print(df)  

x = np.linspace(-1, 1, 100)
y = true_function(x)


plt.figure(figsize=(8, 5))  

plt.plot(x, y, label='y = 10 sin(0.8πx)')
plt.scatter(x_random, y_random, color="red", label="観測点")

plt.legend()
plt.savefig("ex1.2.png")
plt.show()