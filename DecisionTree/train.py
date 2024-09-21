import numpy as np
import os
import pandas as pd
from decision_tree import DecisionTree, Node

path = os.getcwd()
p_train = os.path.join(path, "DecisionTree", "car", "train.csv")
train_df = pd.read_csv(p_train, header=None)

train_data = train_df.values

p_test = os.path.join(path, "DecisionTree", "car", "test.csv")
test_df = pd.read_csv(p_test, header=None)

test_data = test_df.values

X_train, y_train = train_data[:,:-1], train_data[:, -1]
X_test, y_test = test_data[:,:-1], test_data[:, -1]

clf = DecisionTree(max_depth=6)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
test_preds = clf.predict(X_train)
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

test_acc = accuracy(y_train, test_preds)
acc = accuracy(y_test, predictions)
print(test_acc)
print(acc)
