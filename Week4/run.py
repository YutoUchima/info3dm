import datasets
X, Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial2_features(X)
print(ex_X)
print(Y)


import regression

model = regression.RidgeRegression(alpha=0)
print(model.alpha)


model.fit(ex_X, Y)
print(model.theta)