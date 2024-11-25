import numpy as np
import pandas as pd

def load_data():
    train_data = pd.read_csv('bank-note/train.csv', header=None)
    test_data = pd.read_csv('bank-note/test.csv', header=None)
    
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    print(X_train)
