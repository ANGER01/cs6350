from scipy.optimize import minimize
import numpy as np
import pandas as pd

# Load and preprocess data
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

# Define the dual SVM objective function
def dual_svm_obj(alpha, X, y, C):
    m, n = X.shape
    K = np.dot(X, X.T)  # Linear kernel
    return 0.5 * np.dot(alpha.T, np.dot(K, alpha)) - np.sum(alpha)

# Define the constraints: sum(alpha_i * y_i) = 0 and 0 <= alpha_i <= C
def dual_svm_constraints(alpha, y):
    return np.dot(alpha, y)  # sum(alpha_i * y_i) = 0

# SVM dual training function
def train_dual_svm(X_train, y_train, C):
    m, n = X_train.shape
    alpha_init = np.zeros(m)
    constraints = {'type': 'eq', 'fun': dual_svm_constraints, 'args': (y_train,)}
    bounds = [(0, C)] * m
    
    result = minimize(dual_svm_obj, alpha_init, args=(X_train, y_train, C), bounds=bounds, constraints=constraints, method='SLSQP')
    
    alpha_opt = result.x
    support_vectors = alpha_opt > 1e-5
    w = np.dot(X_train[support_vectors].T, y_train[support_vectors] * alpha_opt[support_vectors])
    
    # Compute the bias term b
    b = np.mean(y_train[support_vectors] - np.dot(X_train[support_vectors], w))
    
    return w, b, alpha_opt

# Main code to run the dual SVM
X_train, y_train, X_test, y_test = load_data("bank-note/train.csv", "bank-note/test.csv")
C_values = [100 / 873, 500 / 873, 700 / 873]
for C in C_values:
    w, b, alpha_opt = train_dual_svm(X_train, y_train, C)

    # Predict using the learned weights
    def predict_dual_svm(X, w, b):
        return np.sign(np.dot(X, w) + b)

    y_pred_train = predict_dual_svm(X_train, w, b)
    y_pred_test = predict_dual_svm(X_test, w, b)

    # Evaluate model
    train_error = np.mean(y_pred_train != y_train)
    test_error = np.mean(y_pred_test != y_test)

    print(f"Train error: {train_error}")
    print(f"Test error: {test_error}")
    print(f"Weights:{w} Bias:{b}")

# Define the Gaussian (RBF) kernel function
def gaussian_kernel(X, gamma):
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    return np.exp(-sq_dists / (2 * gamma ** 2))

# Modify the dual SVM objective to use the kernel
def dual_svm_obj_kernel(alpha, X, y, C, gamma):
    K = gaussian_kernel(X, gamma)
    return 0.5 * np.dot(alpha.T, np.dot(K, alpha)) - np.sum(alpha)

# Train the SVM using the Gaussian kernel
def train_dual_svm_kernel(X_train, y_train, C, gamma):
    m, n = X_train.shape
    alpha_init = np.zeros(m)
    constraints = {'type': 'eq', 'fun': dual_svm_constraints, 'args': (y_train,)}
    bounds = [(0, C)] * m
    
    # Optimize the dual objective
    result = minimize(dual_svm_obj_kernel, alpha_init, args=(X_train, y_train, C, gamma), 
                      bounds=bounds, constraints=constraints, method='SLSQP')
    alpha_opt = result.x
    
    # Identify support vectors
    support_vector_indices = alpha_opt > 1e-5
    support_vectors = X_train[support_vector_indices]
    support_labels = y_train[support_vector_indices]
    support_alphas = alpha_opt[support_vector_indices]
    
    # Compute the bias term b
    b = np.mean(
        support_labels - np.sum(
            support_alphas * support_labels[:, None] * 
            gaussian_kernel(support_vectors, gamma),
            axis=0
        )
    )
    
    return alpha_opt, b, support_vector_indices


# Predict function for kernel SVM
def predict_dual_svm_kernel(X, support_vectors, support_labels, support_alphas, b, gamma):
    # Compute the kernel matrix between test samples and support vectors
    K = np.exp(-np.sum((X[:, None] - support_vectors[None, :]) ** 2, axis=2) / (2 * gamma ** 2))
    
    # Compute the decision function
    decision = np.dot(K, support_alphas * support_labels) + b
    
    # Return the predicted labels
    return np.sign(decision)

# Test with different values of C and gamma
gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100 / 873, 500 / 873, 700 / 873]
t_error = 0
for C in C_values:
    print("STARTING DIFFERENT C")
    for gamma in gamma_values:
        print("TRYING NEW GAMMA VALUE")
        alpha_opt, b, support_vector_indices = train_dual_svm_kernel(X_train, y_train, C, gamma)
        
        # Extract support vectors and corresponding parameters
        support_vectors = X_train[support_vector_indices]
        support_labels = y_train[support_vector_indices]
        support_alphas = alpha_opt[support_vector_indices]
        
        # Evaluate on training data
        y_pred_train = predict_dual_svm_kernel(X_train, support_vectors, support_labels, support_alphas, b, gamma)
        train_error = np.mean(y_pred_train != y_train)
        
        # Evaluate on test data
        y_pred_test = predict_dual_svm_kernel(X_test, support_vectors, support_labels, support_alphas, b, gamma)
        test_error = np.mean(y_pred_test != y_test)
        
        print(f"Train Error: {train_error}, Test Error: {test_error}")
        print(f"Support Vectors:{len(support_vectors)}:{support_vectors}")

