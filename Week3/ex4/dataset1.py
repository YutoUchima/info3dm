# フォント設定
import matplotlib
import matplotlib.font_manager as font_manager
font_path = '/Library/Fonts/Arial Unicode.ttf'
font_prop = font_manager.FontProperties(fname = font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

# ex1
import numpy as np
import matplotlib.pyplot as plt
def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10
    
x = 0
y = true_function(x)
print(x, y)
x = np.linspace(-1, 1, 100)
y = true_function(x)
plt.plot(x, y, label='y = 10 sin(0.8πx)')
plt.legend()
plt.savefig("ex1.1.png")
plt.show()


# ex2
import pandas as pd
np.random.seed(0)
n = 20
x_random = np.random.uniform(-1, 1, n)
y_random = true_function(x_random)
df = pd.DataFrame({
    "観測点": x_random,
    "真値": y_random
})
print(df)  

plt.figure(figsize=(8, 5))  
plt.plot(x, y, label='y = 10 sin(0.8πx)')
plt.scatter(x_random, y_random, color="red", label="観測点")
plt.legend()
plt.savefig("ex1.2.png")
plt.show()


# ex3
noise = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=n) / 2
observed_value = y_random + noise
df["観測値"] = observed_value
print(df)
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='y = 10 sin(0.8πx)')
plt.scatter(x_random, y_random, color="red", label="真値")
plt.scatter(x_random, observed_value, color="green", label="観測値")
plt.legend()
plt.savefig("ex1.3.png")
plt.show()


# ex4
df.to_csv("ex1.4.tsv", sep="\t", index=False)