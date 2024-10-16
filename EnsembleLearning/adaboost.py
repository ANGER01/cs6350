from DecisionTree.decision_tree import Tree
import os
import numpy as np
import pandas as pd
from Helpers.helpers import get_data

Bank_X_train, Bank_y_train, Bank_X_test, Bank_y_test = get_data("bank")

def create_stump(x, y) -> Tree:
    """Takes cleaned data(Not Numerical, only catagorical)

    Args:
        x (ndarray): pandas data frame with feature values
        y (ndarray): pandas series with labels

    Returns:
        DecisionTree: instance of decision tree
    """
    stump = Tree(max_depth=1)
    stump.fit(x, y, func="Entropy")
    return stump

