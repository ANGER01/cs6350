from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np


path = os.getcwd()
p_train = os.path.join(path, "DecisionTree", "bank", "train.csv")
train_df = pd.read_csv(p_train, header=None)

train_data = train_df.values

p_test = os.path.join(path, "DecisionTree", "bank", "test.csv")
test_df = pd.read_csv(p_test, header=None)

test_data = test_df.values

X_train, y_train = train_data[:,:-1], train_data[:, -1]
X_test, y_test = test_data[:,:-1], test_data[:, -1]


for i in range(0, 6):
    print(i)
"""cols = test_df.columns

for col in cols:
    if test_df[col].dtype != "O":
        median = test_df[col].median()
        test_df[col] = test_df[col].apply((lambda x: 'above' if x >= median else 'below'))
    
for col in cols:
    print(test_df[col].dtype)"""
    

"""all_splits = []
items = np.unique(X_column)
print(items)
print("::::::::::::::::")
for item in items:
    all_splits.append(np.argwhere(X_column == item))

for split in all_splits:
    print(split)
    print("----------")"""

def information_gain(y, X_column):
    # parent entropy
    parent_entropy = entropy(y)
    # create children
    children = split(X_column)
    """    for child in children:
        print("THE CHILD'S INDICES",child)"""
    n = len(y)    
    if len(children) == 0:
        return 0
    
    weighted_entropy = 0
    # calculate the weighted avg. entropy of children
    for child in children:
        child_length = len(child)
        child_entropy = entropy(y[child])
        weighted_entropy += (child_length/n) * child_entropy
    # calculate the IG
    information_gain = parent_entropy - weighted_entropy
    return information_gain

def best_split( X, y, feat_idxs):
    best_gain = -1
    split_idx = None
    for feat_idx in feat_idxs:
        X_column = X[:,feat_idx]
        gain = information_gain(y, X_column)
        if gain > best_gain:
            best_gain = gain
            split_idx = feat_idx
    return split_idx

def split(X_column):
    all_splits = []
    items = np.unique(X_column)
    for item in items:
        all_splits.append(np.argwhere(X_column == item).flatten())
    return all_splits

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    ps = counts / counts.sum()
    return -np.sum((ps * np.log2(ps + 1e-9)))


