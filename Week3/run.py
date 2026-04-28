import datasets 

X,Y = datasets.load_linear_example1()
print(X)
print(X[0])
print(Y)


#ver2 test
import importlib
import regression 
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X, Y)
print(model.theta)



# ver3 test
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X, Y)
print(model.predict(X))



# ver4 test
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X, Y)
print(model.score(X, Y))