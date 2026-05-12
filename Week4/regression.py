import numpy as np

# ver1
class LinearRegression:
    x = None
    theta = None
    y = None

# ver2
    def fit(self, x, y):
        temp = np.linalg.inv(np.dot(x.T, x))
        self.theta = np.dot(np.dot(temp, x.T), y)

# ver3
    def predict(self, x):
        return np.dot(x, self.theta)
    

# ver4
    def score(self, x, y):
        error = self.predict(x) - y
        return (error**2).sum()

class RidgeRegression(LinearRegression):
    alpha = None
# ver5
    def __init__(self, alpha=0.1):
        self.alpha = alpha

# ver6
    def fit(self, input, output):
        xTx = np.dot(input.T, input)
        I = np.eye(len(xTx))
        self.theta = np.dot(np.dot(np.linalg.inv(xTx + self.alpha * I), input.T), output)

