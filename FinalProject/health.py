import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("heart_disease_uci.csv")

    # Preprocess data
    df.drop(columns=['id'], inplace=True)
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric NaN with column medians
    for column in df.select_dtypes(include=['object', 'bool']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)  # Fill categorical/boolean NaN with mode

    # Encode categorical columns
    encoder = LabelEncoder()
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = encoder.fit_transform(df[column])

    # Check for missing values after preprocessing
    print(f"Missing values after preprocessing: {df.isna().sum().sum()}")

    # Separate features (X) and labels (y)
    X = df.iloc[:, :-1]  # All rows, all columns except the last
    y = df.iloc[:, -1]   # All rows, only the last column

    # Convert to binary classification: 0 remains 0, all other values become 1
    y = (y > 0).astype(int)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Base Decision Tree
    clf = DecisionTreeClassifier(
        max_depth=2, 
        random_state=42,
        class_weight={0: 1, 1: 2}  # Penalize false negatives more
    )

    def print_per_class_accuracy(y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n{model_name} results:")
             
        # Calculate and print false negative rate
        false_negative_rate = cm[1,0] / (cm[1,0] + cm[1,1])
        false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1])
        print(f"\nFalse Negative Rate: {false_negative_rate:.2%}")
        print(f"False Positive Rate: {false_positive_rate:.2%}")

    # Decision Tree
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nDecision Tree Accuracy: {accuracy:.2f}")
    print_per_class_accuracy(y_test, y_pred, "Decision Tree")

    # Bagged Trees
    bagged_clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2,
            class_weight={0: 1, 1: 2}
        ),
        n_estimators=50,
        random_state=42
    )
    bagged_clf.fit(X_train, y_train)
    y_pred_bagged = bagged_clf.predict(X_test)
    accuracy_bagged = accuracy_score(y_test, y_pred_bagged)
    print(f"\nBagged Trees Accuracy: {accuracy_bagged:.2f}")
    print_per_class_accuracy(y_test, y_pred_bagged, "Bagged Trees")

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=50, 
        max_depth=2, 
        random_state=42,
        class_weight={0: 1, 1: 2}
    )
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\nRandom Forest Accuracy: {accuracy_rf:.2f}")
    print_per_class_accuracy(y_test, y_pred_rf, "Random Forest")

    # AdaBoost
    ada_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=1,
            class_weight={0: 1, 1: 2}
        ),
        n_estimators=50,
        random_state=42
    )
    ada_clf.fit(X_train, y_train)
    y_pred_ada = ada_clf.predict(X_test)
    accuracy_ada = accuracy_score(y_test, y_pred_ada)
    print(f"\nAdaBoost Accuracy: {accuracy_ada:.2f}")
    print_per_class_accuracy(y_test, y_pred_ada, "AdaBoost")
