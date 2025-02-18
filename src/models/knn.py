import numpy as np
import pandas as pd
from Collections import Counter 

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        
        # compute the distance
        distances = [self._distance(x, x_train) for x_train in self.X_train]

        # get closest k
        idxs = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in idxs]

        # predict using majority vote
        most_common = Counter(k_nearest_labels).most_common()[0][0]
        return most_common

    def _distance(self, x1, x2, type, p = 1):
        """
        params:
            type: str
                This determines the method used to calculate the distance between any two points. It can either be 'manhattan', 'euclidean' or 'minkowski' distance
            p: int
                This determines the power used for minkowski distance

        """
        if type == "euclidean":
            p = 2
        
        if type == "minkowski" and p < 3:
            assert ValueError("Power value not passed or a value < 3 passed.")
 
        dist = (np.sum((x1-x2)**p))**(1/p)
        return dist
