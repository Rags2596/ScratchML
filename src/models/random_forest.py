import numpy as np
import random
from collections import Counter

def gini_impurity(y):
    """Calculate gini impurity for the given subset of y values"""

    counts = np.bincount(y)
    gini = 0
    for i in counts:
        gini += (i/sum(counts))**2
    return 1-gini

def entropy(y):
    counts = np.bincount(y)
    entropy_val = 0
    for i in counts:
        entropy_val += -(i/sum(counts) * np.log2(i/sum(counts)))
    
    return entropy_val


class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None



class DecisionTree:
    def __init__(self, min_samples_split = 2, n_features = None, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features 
        self.root = None
    
    def fit(self, X, y, depth = 0):
        """
        Build a tree
        """
        self.n_features = X.shape[1] if not self.n_features else min(X,shape(1), self.n_features)
        self.root = self._grow_tree(X, y)



    def _grow_tree(self, X, y, depth = 0):
        n_samples, n_feats = X.shape
        n_labels = len(set(y))

        #Check stopping criteria
        if depth>=self.max_depth or n_labels == 1 or n_samples<self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)

        # find the best split
        feature_indices = np.random.choice(n_feats, self.n_features, replace = False)
        best_feature, best_thresh = self._best_split(X, y, feature_indices)

        # create child nodes
        left_idxs, right_idxs = np.argwhere(X[:, best_feature] <= best_thresh).flatten(), np.argwhere(X[:, best_feature] > best_thresh).flatten()
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feature_indices):
        """
        params:
            X: features data
            y: associated classes
            feature_indices: random selection of features to be used to train a tree
        """

        best_gini = np.inf
        best_feature, best_threshold = None, None

        for feat_idx in feature_indices:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_indices = np.argwhere(X_column <= threshold).flatten()
                right_indices = np.argwhere(X_column > threshold).flatten()

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue # if split leads to empty sets on either sides, its deemed as a bad split

                gini_left = gini_impurity(y[left_indices])
                gini_right = gini_impurity(y[right_indices])

                weighted_gini = len(left_indices)/len(y) * gini_left + len(right_indices)/len(y) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_threshold = threshold
                    best_feature = feat_idx
        return best_feature, best_threshold


    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _predict_single(self, x, node):
        """Predict for a single sample"""
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        """Predict for multiple samples"""
        return np.array([self._predict_single(x, self.root) for x in X])
        

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split = 2, n_features = None, max_depth=100):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features =  n_features
        self.trees = []

    def fit(self, X, y):
        """Train multiple trees on bootstrapped datasets"""
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth, min_samples_split = self.min_samples_split, n_features = self.n_features)

            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace = True)
        return X[idxs], y[idxs]


    def predict(self, X):
        """Aggregate predictions from all trees (majority voting)"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions)


