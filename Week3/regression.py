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




