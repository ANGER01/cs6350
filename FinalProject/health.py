import pandas as pd
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":

    df = pd.read_csv("heart_disease_uci.csv")
    #preprocess
    print(df.shape)
    encoder = LabelEncoder()
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = encoder.fit_transform(df[column])
    # Separate features (X) and labels (y)
    print(df["thal"].isna().sum())
    X = df.iloc[:, :-1]  # All rows, all columns except the last
    y = df.iloc[:, -1]   # All rows, only the last column

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)