import numpy as np
import os
import pandas as pd
from decision_tree import Tree

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns

    for col in cols:
        if df[col].dtype != "O":
            median = df[col].median()
            df[col] = df[col].apply((lambda x: 'above' if x >= median else 'below'))
    return df

def doubly_clean(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    for col in cols:
        if df[col].mode()[0] != 'unknown':
            most_common_value = df[col].mode()[0]
        else:
            most_common_value = df[col].mode()[1]
        df[col] = df[col].replace('unknown', most_common_value)
    return

def get_data(folder: str):
    path = os.getcwd()
    p_train = os.path.join(path, folder, "train.csv")
    train_df = pd.read_csv(p_train, header=None)
    train_df = clean(train_df)
    train_data = train_df.values

    p_test = os.path.join(path, folder, "test.csv")
    test_df = pd.read_csv(p_test, header=None)
    test_df = clean(test_df)
    test_data = test_df.values

    X_train, y_train = train_data[:,:-1], train_data[:, -1]
    X_test, y_test = test_data[:,:-1], test_data[:, -1]
    
    return X_train, y_train, X_test, y_test

def result(max_depth, X_train, y_train, X_test, y_test):
    for i in range(1,max_depth + 1):
        clf = Tree(max_depth=i)
        clf.fit(X_train, y_train)

        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)

        train_acc = accuracy(y_train, train_pred)
        test_acc = accuracy(y_test, test_pred)
        print("Max Depth: ", i)
        print("Accuracy of Train data", train_acc)
        print("Accuracy of Test", test_acc)
        print("\n----------\n")
        
# Car_X_train, Car_y_train, Car_X_test, Car_y_test = get_data("car")
Bank_X_train, Bank_y_train, Bank_X_test, Bank_y_test = get_data("bank")

# car_max_depth = Car_X_train.shape[1]
bank_max_depth = Bank_X_train.shape[1]

print(Bank_X_train.shape)
#print(Bank_X_train)
#result(car_max_depth, Car_X_train, Car_y_train, Car_X_test, Car_y_test, func="Entropy")
#result(car_max_depth, Car_X_train, Car_y_train, Car_X_test, Car_y_test, func="Gini")
#result(car_max_depth, Car_X_train, Car_y_train, Car_X_test, Car_y_test, func="Majority Error")

result(bank_max_depth, Bank_X_train, Bank_y_train, Bank_X_test, Bank_y_test)
#result(bank_max_depth, Bank_X_train, Bank_y_train, Bank_X_test, Bank_y_test, func="Gini")
#result(bank_max_depth, Bank_X_train, Bank_y_train, Bank_X_test, Bank_y_test, func="Majority Error")
