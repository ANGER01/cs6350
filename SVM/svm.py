import numpy as np
import pandas as pd

def normalize_features(X_train, X_test):
    """Normalize features using min-max scaling"""
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    
    for i in range(X_train.shape[1]):
        min_val = X_train[:, i].min()
        max_val = X_train[:, i].max()
        if max_val - min_val != 0:
            X_train_norm[:, i] = (X_train[:, i] - min_val) / (max_val - min_val)
            X_test_norm[:, i] = (X_test[:, i] - min_val) / (max_val - min_val)
    
    return X_train_norm, X_test_norm

def load_data():
    train_data = pd.read_csv('bank-note/train.csv', header=None)
    test_data = pd.read_csv('bank-note/test.csv', header=None)
    
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    return X_train, y_train, X_test, y_test


def add_regularization(w, subgradient_w):
    """
    The total loss :( 1/2 * ||w||^2 + Hingle_loss) has w term to be added after getting subgradient of 'w'
    
      total_w = regularization_term + subgradient_term
    i.e total_w = w + C *  ∑ (-y*x)
    
    """
    return w + subgradient_w


def subgradients(x, y, w, b, C):
    """
    :x: inputs [[x1,x2], [x2,x2],...]
    :y: labels [1, -1,...]
    :w: initial w
    :b: initial b
    :C: tradeoff/ hyperparameter
    
    """
    # Initialize subgradients
    subgrad_w = w  # Changed: Initialize with w for regularization term
    subgrad_b = 0
    
    # Calculate decision value
    f_xi = np.dot(w.T, x) + b
    decision_value = y * f_xi

    # Update subgradients based on hinge loss condition
    if decision_value < 1:
        subgrad_w = subgrad_w + C * (-y * x)  # Changed: Add to regularization term
        subgrad_b = -C * y
    
    return (subgrad_w, subgrad_b)  # Removed: add_regularization() call as regularization is now included


def stochastic_subgrad_descent(x_vals: np.array, y_vals: np.array, int_weights, int_bias, C,gamma_0, a, T=100):
    """
    Simple stochastic gradient descent implementation
    """
    w = int_weights
    b = int_bias
    n_samples = len(y_vals)
    
    # Create indices array
    indices = np.arange(n_samples)
    
    best_w = w
    best_b = b
    best_error = float('inf')
    
    for t in range(1, T+1):
        # Shuffle data at the start of each epoch
        np.random.shuffle(indices)
        
        # Calculate learning rate
        learning_rate = gamma_0 / (1 + (gamma_0 / a) * (t * n_samples))
        
        # Process each sample
        for idx in indices:
            x = x_vals[idx]
            y = y_vals[idx]
            
            # Get subgradients for single sample
            w_grad, b_grad = subgradients(x, y, w, b, C)
            
            # Update weights and bias
            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad
        
        # Keep track of best model
        current_error = calculate_error(x_vals, y_vals, w, b)
        if current_error < best_error:
            best_error = current_error
            best_w = w.copy()
            best_b = b
    
    return best_w, best_b


def predict(x, w, b):
    """
    Predict class using w and b
    Returns 1 if w·x + b ≥ 0, -1 otherwise
    """
    temp = np.dot(w, x) + b
    # print(f"temp: {temp}")
    return 1 if temp >= 0 else -1

def calculate_error(X, y, w, b):
    """
    Calculate prediction error rate
    """
    incorrect = 0
    total = len(y)
    
    for i in range(total):
        prediction = predict(X[i], w, b)
        # Convert 0 labels to -1 for comparison
        actual = 1 if y[i] == 1 else -1
        print(f"Predicted: {prediction}")
        if prediction != actual:
            incorrect += 1
            
    return incorrect / total

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    
    # Normalize features
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    
    # Try different C values
    C_values = [(100/875), (500/875), (700/875)]
    best_C = None
    best_train_error = float('inf')
    best_test_error = float('inf')
    best_model = None
    
    initial_weights = np.zeros(X_train.shape[1])  # Initialize with zeros
    initial_bias = 0
    
    for C in C_values:
        w, b = stochastic_subgrad_descent(X_train_norm, y_train, initial_weights, initial_bias, C, .01, 1, T=100)
        
        train_error = calculate_error(X_train_norm, y_train, w, b)
        test_error = calculate_error(X_test_norm, y_test, w, b)
        
        print(f"C={C}: Training error: {train_error:.4f}, Test error: {test_error:.4f}")
        
        if test_error < best_test_error:
            best_C = C
            best_test_error = test_error
            best_train_error = train_error
            best_model = (w, b)
    
    print(f"\nBest model (C={best_C}):")
    print(f"Training error: {best_train_error:.4f}")
    print(f"Test error: {best_test_error:.4f}")
    print(best_model[0])