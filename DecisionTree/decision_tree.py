import numpy as np
from collections import Counter
import pandas as pd

class Node:
    def __init__(self, feature=None, children=None,depth=0,*,value=None):
        self.feature = feature
        self.choice = None
        self.children = children
        self.value = value
        self.depth = depth
        
    def is_leaf_node(self):
        return self.value is not None


class Tree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root: Node=None

    def fit(self, X, y, func):
        f = self.func_chooser(func)
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y, f)

    def func_chooser(self, func: str):
        if func.lower() == "gini":
            return self._gini_index
        elif (func.lower() == "majority error") or (func.lower() == "me"):
            return self._majority_error
        else:
            return self._entropy
    
    def _grow_tree(self, X, y, func, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(self.n_features, n_feats, replace=False)

        # find the best split
        best_feature = self._best_split(X, y, feat_idxs, func=func)
        # create child nodes
        child_indices = self._split(X[:, best_feature])
        # for each unique value in column_X grow tree?
        kids = []
        for index in child_indices:
            choices = X[index,:]
            choice = choices[0][best_feature]
            kids.append((self._grow_tree(choices, y[index], func, depth+1),choice))
            
        return Node(best_feature, kids, depth=depth)

    def _best_split(self, X, y, feat_idxs, func):
        best_gain = -1
        split_idx = None
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            gain = self._information_gain(y, X_column, func=func)
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
        return split_idx

    def _information_gain(self, y, X_column, func):
        # parent entropy
        parent_entropy = func(y)
        # create children
        children = self._split(X_column)
        n = len(y)    
        if len(children) == 0:
            return 0

        weighted_entropy = 0
        # calculate the weighted avg. entropy of children
        for child in children:
            child_length = len(child)
            child_entropy = func(y[child])
            weighted_entropy += (child_length/n) * child_entropy
        # calculate the IG

        information_gain = parent_entropy - weighted_entropy
        return information_gain

    def _split(self, X_column):
        all_splits = []
        items = np.unique(X_column)
        for item in items:
            all_splits.append(np.argwhere(X_column == item).flatten())
        return all_splits

    def _gini_index(self, y) -> float:
        _, counts = np.unique(y, return_counts=True)
        ps = counts / counts.sum()
    
        return 1 - (sum(ps**2))

    def _majority_error(self, y) -> float:
        _, counts = np.unique(y, return_counts=True)
        ps = counts / counts.sum()
        return np.min(ps)

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        ps = counts / counts.sum()
        return -np.sum((ps * np.log2(ps + 1e-9)))


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node: Node):
        if node.is_leaf_node():
            return node.value

        for child in node.children:
            if(child[1] == x[node.feature]):
                return self._traverse_tree(x, child[0])
