import numpy as np
import pandas as pd

def unit_step_func(x):
    return np.where(x > 0 , 1, 0)

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0 , 1, 0)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.learning_rate * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    def predict(self, X: np.ndarray):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_func(linear_output)

class VotedPerceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_func = unit_step_func
        self.weights_list = []
        self.bias_list = []
        self.count_list = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        current_weights = np.zeros(n_features)
        current_bias = 0
        correct_count = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, current_weights) + current_bias
                y_predicted = self.activation_func(linear_output)

                if y_[idx] != y_predicted:
                    if correct_count > 0:
                        self.weights_list.append(current_weights.copy())
                        self.bias_list.append(current_bias)
                        self.count_list.append(correct_count)
                    
                    update = self.learning_rate * (y_[idx] - y_predicted)
                    current_weights = current_weights + update * x_i
                    current_bias = current_bias + update
                    correct_count = 0
                else:
                    correct_count += 1

        if correct_count > 0:
            self.weights_list.append(current_weights)
            self.bias_list.append(current_bias)
            self.count_list.append(correct_count)

    def predict(self, X: np.ndarray):
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            weighted_votes = 0
            total_votes = 0
            for weights, bias, count in zip(self.weights_list, self.bias_list, self.count_list):
                linear_output = np.dot(X[i], weights) + bias
                prediction = self.activation_func(linear_output)
                weighted_votes += count * prediction
                total_votes += count
            predictions[i] = 1 if weighted_votes > total_votes/2 else 0
        return predictions

class AveragedPerceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
        self.averaged_weights = None
        self.averaged_bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.averaged_weights = np.zeros(n_features)
        self.averaged_bias = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                if y_[idx] != y_predicted:
                    update = self.learning_rate * (y_[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update

                self.averaged_weights += self.weights
                self.averaged_bias += self.bias

        total_updates = self.n_iterations * n_samples
        self.averaged_weights /= total_updates
        self.averaged_bias /= total_updates

    def predict(self, X: np.ndarray):
        linear_output = np.dot(X, self.averaged_weights) + self.averaged_bias
        return self.activation_func(linear_output)

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
    
    perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
    perceptron.fit(X_train, y_train)
    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Standard Perceptron Test Accuracy: {accuracy:.4f}")
    print(perceptron.weights)
    
    voted_perceptron = VotedPerceptron(learning_rate=0.5, n_iterations=10)
    voted_perceptron.fit(X_train, y_train)
    voted_predictions = voted_perceptron.predict(X_test)
    voted_accuracy = np.mean(voted_predictions == y_test)
    print(f"\nVoted Perceptron Test Accuracy: {voted_accuracy:.4f}")
    
    print("\nVoted Perceptron Weight Vectors and Counts:")
    for weights, count in zip(voted_perceptron.weights_list, voted_perceptron.count_list):
        print(f"{weights} : {count}")
    
    averaged_perceptron = AveragedPerceptron(learning_rate=0.1, n_iterations=10)
    averaged_perceptron.fit(X_train, y_train)
    averaged_predictions = averaged_perceptron.predict(X_test)
    averaged_accuracy = np.mean(averaged_predictions == y_test)
    print(f"\nAveraged Perceptron Test Accuracy: {averaged_accuracy:.4f}")
    print(averaged_perceptron.weights)
