from Model_Trainer import train_model
import torch.nn as nn
import torch

class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.value = value  # Class label for leaf node
        self.left = left  # Left child node
        self.right = right  # Right child node


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _gini(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _split_data(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        return X_left, y_left, X_right, y_right

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_index, threshold)

                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)

                gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            _, class_counts = np.unique(y, return_counts=True)
            value = np.argmax(class_counts)
            return Node(value=value)

        feature_index, threshold = self._best_split(X, y)
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature_index, threshold)

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature_index=feature_index, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def predict(self, X):
        return [self._traverse(x, self.root) for x in X]