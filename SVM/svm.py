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
    for x_i, y_i in zip(x,y):
        f_xi = np.dot(w.T, x_i) + b

        decision_value = y_i * f_xi

        if decision_value < 1:
            subgrad_w += - y_i*x_i
            subgrad_b += -1 * y_i
        else:
            subgrad_w += 0
            subgrad_b += 0
    
    # multiply by C after summation of all subgradients for a given samples of x,y
    subgrad_w = C * subgrad_w
    subgrad_b = C * subgrad_b
    return (add_regularization(w, subgrad_w), subgrad_b)


def stochastic_subgrad_descent(data, initial_values, B, C, T=1):
    """
    :data: Pandas data frame
    :initial_values: initialization for w and b
    :B: sample size for random data selection
    :C: hyperparameter, tradeoff between hard margin and hinge loss
    :T: # of iterations
    
    """
    w, b = initial_values
    for t in range(1, T+1):
        
        # randomly select B data points 
        training_sample = data.sample(B)
        
        # set learning rate
        learning_rate = 1/t
        
        # prepare inputs in the form [[h1, w1], [h2, w2], ....]
        x = training_sample[['height', 'weight']].values
      
        # prepare labels in the form [1, -1, 1, 1, - 1 ......]
        y = training_sample['gender'].values
      
        sub_grads = subgradients(x,y, w, b, C)
        
        # update weights
        w = w - learning_rate * sub_grads[0]
        
        # update bias
        b = b - learning_rate * sub_grads[1]
    return (w,b)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    print(X_train)
