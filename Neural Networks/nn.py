import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1, zero_flag=False):
        """
        Initialize the neural network.
       
        Parameters:
        - layer_sizes: List of integers, specifying the size of each layer (e.g., [input, hidden1, hidden2, output]).
        - learning_rate: Learning rate for stochastic gradient descent.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.zero_flag = zero_flag
        self.weights = []
        self.biases = []

        if not self.zero_flag:
            for i in range(len(layer_sizes) - 1):
                self.weights.append(np.random.normal(0, 1, (layer_sizes[i], layer_sizes[i+1])))
                self.biases.append(np.random.normal(0, 1, (1, layer_sizes[i+1])))
        else:
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
        initial_lr = self.learning_rate
        d = 2  # You can adjust this parameter
        
        for epoch in range(epochs):
            # Update learning rate according to schedule
            self.learning_rate = initial_lr / (1 + (initial_lr * epoch / d))
            
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            activations = self.forward(X_shuffled)
            self.backward(activations, y_shuffled)
        
        # Calculate final training loss and accuracy
        final_activations = self.forward(X)
        training_loss = np.mean((final_activations[-1] - y) ** 2)
        training_predictions = (final_activations[-1] > 0.5).astype(int)
        training_accuracy = np.sum(y.flatten() == training_predictions.flatten()) / y.shape[0]
        return training_loss, training_accuracy

    def predict(self, X):
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

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data("bank-note/train.csv", "bank-note/test.csv")

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    zero_flag = False
    
    if zero_flag:
        print("Weights Zeroed out")
    else:
        print("Weights random normal distribution")
    widths = [5, 10, 25, 50, 100]
    for val in widths:
        nn = NeuralNetwork(layer_sizes=[4, val, val, 1], learning_rate=0.4, zero_flag=zero_flag)
        train_loss, train_accuracy = nn.train(X_train, y_train, epochs=150)

        predictions = nn.predict(X_test)
        predictions = (predictions > 0.5).astype(int)
        
        correct = np.sum(y_test.flatten() == predictions.flatten())
        test_accuracy = correct / y_test.shape[0]

        print(f"Layer Width:{val}")
        print(f"Training Loss:{train_loss:.4f}")
        print(f"Training Accuracy:{train_accuracy:.4f}")
        print(f"Test Accuracy:{test_accuracy:.4f}")
        print("------------------------")
