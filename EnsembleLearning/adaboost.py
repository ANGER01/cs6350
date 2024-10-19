import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from Helpers.helpers import get_data, result, accuracy
import numpy as np
import pandas as pd
from DecisionTree.decision_tree import Tree

def adaboost_clf(X_train, Y_train, X_test, Y_test, T, clf: Tree):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    for i in range(T):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, weights= w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * alpha_m for x in pred_test_i])]
    
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), get_error_rate(pred_test, Y_test)
           
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

X_train, y_train, X_test, y_test = get_data("bank")

print(X_train.shape)
train_accuracy = []
test_accuracy = []

for T in range(1,501):
    temp_train, temp_test = adaboost_clf(X_train, y_train, X_test, y_test, T, Tree(max_depth=1))
    train_accuracy.append(temp_train)
    test_accuracy.append(temp_test)
    print("YOWZA")

print(train_accuracy)
print(test_accuracy)