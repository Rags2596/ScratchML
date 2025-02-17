import numpy as np

class LogisticRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_predictions)

            dw = (1/n_samples) * 2*np.dot(X.T, (predictions - y)) # grdient with respect to weights
            db = (1/n_samples) * 2*np.sum(predictions - y)# grdient with respect to bias

            # update weights and bias

            self.weights = self.weights - dw*self.lr
            self.bias = self.bias - db*self.lr


    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_predictions)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred


    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))