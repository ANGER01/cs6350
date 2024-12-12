import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        """
        Initialize the neural network.
       
        Parameters:
        - layer_sizes: List of integers, specifying the size of each layer (e.g., [input, hidden1, hidden2, output]).
        - learning_rate: Learning rate for stochastic gradient descent.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.zeros((layer_sizes[i], layer_sizes[i+1])))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        """Derivative of the sigmoid function applied to the activation directly."""
        return a * (1 - a)

    def forward(self, X):
        """Perform a forward pass through the network."""
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            a = self.sigmoid(z)
            activations.append(a)
        return activations

    def backward(self, activations, y):
        """Perform a backward pass through the network and update weights and biases."""
        m = y.shape[0]
        deltas = [(activations[-1] - y) * self.sigmoid_derivative(activations[-1])]

        # Backpropagate deltas
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            deltas.append(delta)
        deltas.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m

    def train(self, X, y, epochs=100):
        """
        Train the network using stochastic gradient descent.
       
        Parameters:
        - X: Training data (m x n).
        - y: Training labels (m x output_size).
        - epochs: Number of training iterations.
        """
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(activations, y)

            # Compute loss (mean squared error)
            loss = np.mean((activations[-1] - y) ** 2)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        """
        Predict the output for given input X.
       
        Parameters:
        - X: Input data (m x n).
       
        Returns:
        - Predictions (m x output_size).
        """
        activations = self.forward(X)
        return activations[-1]

def load_data(train_path, test_path):
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)
    
    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    return X_train, y_train, X_test, y_test

# Example usage:
if __name__ == "__main__":
    # Create synthetic training data
    X_train, y_train, X_test, y_test = load_data("bank-note/train.csv", "bank-note/test.csv")

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Define and train the network
    nn = NeuralNetwork(layer_sizes=[4, 4, 64, 1], learning_rate=0.1)
    print(nn.weights)
    nn.train(X_train, y_train, epochs=100)
    print(nn.weights)
    # Test the network
    predictions = nn.predict(X_test)
    predictions = (predictions > 0.5).astype(int)

    
    correct = np.sum(y_test.flatten() == predictions.flatten())
    accuracy = correct / y_test.shape[0]
    # Print results
    print(f"Accuracy:{accuracy}")
    # Print results
    print("Test Predictions:", predictions.flatten())
    print("True Labels:", y_test.flatten())