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
    subgrad_w = 0
    subgrad_b = 0
    
    # sum over all subgradients of hinge loss for a given samples x,y
    f_xi = np.dot(w.T, x) + b

    decision_value = y * f_xi

    if decision_value < 1:
        subgrad_w += - y*x
        subgrad_b += -1 * y
    else:
        subgrad_w += 0
        subgrad_b += 0
    
    # multiply by C after summation of all subgradients for a given samples of x,y
    subgrad_w = C * subgrad_w
    subgrad_b = C * subgrad_b
    return (add_regularization(w, subgrad_w), subgrad_b)


def stochastic_subgrad_descent(x_vals: np.array, y_vals: np.array, int_weights, int_bias, C, T=100, batch_size=32):
    """
    Modified to use mini-batches and learning rate decay
    """
    w = int_weights
    b = int_bias
    n_samples = len(y_vals)
    
    # Create indices array and shuffle
    indices = np.arange(n_samples)
    
    best_w = w
    best_b = b
    best_error = float('inf')
    
    for t in range(1, T+1):
        # Shuffle data at the start of each epoch
        np.random.shuffle(indices)
        
        # Mini-batch processing
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, n_samples)]
            
            # Calculate learning rate with decay
            learning_rate = 1.0 / (1.0 + 0.01 * t)
            
            # Process mini-batch
            grad_w = np.zeros_like(w)
            grad_b = 0
            
            for idx in batch_indices:
                x = x_vals[idx]
                y = y_vals[idx]
                sub_grads = subgradients(x, y, w, b, C)
                grad_w += sub_grads[0]
                grad_b += sub_grads[1]
            
            # Average gradients over mini-batch
            grad_w /= len(batch_indices)
            grad_b /= len(batch_indices)
            
            # Update weights and bias
            w = w - learning_rate * grad_w
            b = b - learning_rate * grad_b
        
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
    return 1 if np.dot(w, x) + b >= 0 else -1

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
        if prediction != actual:
            incorrect += 1
            
    return incorrect / total

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    
    # Normalize features
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    
    # Try different C values
    C_values = [0.1, 1.0, 10.0, 100.0]
    best_C = None
    best_train_error = float('inf')
    best_test_error = float('inf')
    best_model = None
    
    initial_weights = np.zeros(X_train.shape[1])  # Initialize with zeros
    initial_bias = 0
    
    for C in C_values:
        w, b = stochastic_subgrad_descent(X_train_norm, y_train, initial_weights, initial_bias, C, T=100, batch_size=32)
        
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