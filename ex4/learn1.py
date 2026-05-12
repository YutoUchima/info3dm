import dataset1

# ex1.1
dataset1.ex1_1()

# ex1.2
df = dataset1.ex1_2()

# ex1.3
df = dataset1.ex1_3(df)

# ex1.4
dataset1.ex1_4(df)

# ex1.5
loaded_df = dataset1.ex1_5()

print(loaded_df)


# ex1.8
from sklearn.linear_model import LinearRegression

model = LinearRegression()


# ex1.9 途中です
X = df["観測点"].values.reshape(-1, 1)
y = df["観測値"].values
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

model.fit(X_train, y_train)
print("fit完了")