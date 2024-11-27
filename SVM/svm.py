import numpy as np
import pandas as pd

# Load Data
def load_data(train_path, test_path):
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)
    
    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Normalize features
    X_train = 2 * (X_train - X_train.min()) / (X_train.max() - X_train.min()) - 1
    X_test = 2 * (X_test - X_train.min()) / (X_train.max() - X_train.min()) - 1

    # Convert labels to {-1, 1}
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    return X_train, y_train, X_test, y_test

# SVM with Stochastic Subgradient Descent
class PrimalSVM:
    def __init__(self, C, gamma0, a=None, schedule="1+a", max_epochs=100):
        self.C = C
        self.gamma0 = gamma0
        self.a = a
        self.schedule = schedule
        self.max_epochs = max_epochs
        self.w = None
        self.b = 0

    def learning_rate(self, t):
        if self.schedule == "1+a":
            return self.gamma0 / (1 + (self.gamma0 / self.a) * t)
        elif self.schedule == "1+t":
            return self.gamma0 / (1 + t)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        objective_curve = []

        for epoch in range(self.max_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for t, i in enumerate(indices):
                lr = self.learning_rate(t)
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1

                if condition:
                    self.w -= lr * self.w
                else:
                    self.w -= lr * (self.w - self.C * y[i] * X[i])
                    self.b += lr * self.C * y[i]

            # Compute objective function
            hinge_loss = np.maximum(0, 1 - y * (X @ self.w + self.b)).mean()
            objective = 0.5 * np.dot(self.w, self.w) + self.C * hinge_loss
            objective_curve.append(objective)

        return objective_curve

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Evaluate Model
def evaluate(y_true, y_pred):
    return np.mean(y_true != y_pred)

# Main
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data("bank-note/train.csv", "bank-note/test.csv")
    
    Cs = [100 / 873, 500 / 873, 700 / 873]
    gamma0 = .015
    a = 7

    for C in Cs:
        print(f"Testing C={C}")
        
        # Schedule 1
        svm_1 = PrimalSVM(C=C, gamma0=gamma0, a=a, schedule="1+a", max_epochs=100)
        curve_1 = svm_1.fit(X_train, y_train)
        train_error_1 = evaluate(y_train, svm_1.predict(X_train))
        test_error_1 = evaluate(y_test, svm_1.predict(X_test))
        
        # Schedule 2
        svm_2 = PrimalSVM(C=C, gamma0=gamma0, schedule="1+t", max_epochs=100)
        curve_2 = svm_2.fit(X_train, y_train)
        train_error_2 = evaluate(y_train, svm_2.predict(X_train))
        test_error_2 = evaluate(y_test, svm_2.predict(X_test))
        
        print(f"Learned Weights and Bias: W:{svm_1.w} B:{svm_1.b}")
        print(f"Schedule 1+a: Train Error={train_error_1}, Test Error={test_error_1}")
        print(f"Learned Weights and Bias: W:{svm_2.w} B:{svm_2.b}")
        print(f"Schedule 1+t: Train Error={train_error_2}, Test Error={test_error_2}")
