# logistic_regression.bend

def sigmoid(x):
    """
    Compute the sigmoid of x.
    
    Parameters:
    x (float): The input value.
    
    Returns:
    float: The sigmoid of x.
    """
    return 1.0 / (1.0 + exp(-x))

def predict(X, weights):
    """
    Predict the probabilities for input data X using the logistic regression model.
    
    Parameters:
    X (array): Input data matrix.
    weights (array): Model weights.
    
    Returns:
    array: Predicted probabilities.
    """
    return sigmoid(dot(X, weights))

def update_weights(X, y, weights, learning_rate):
    """
    Update the model weights using gradient descent.
    
    Parameters:
    X (array): Input data matrix.
    y (array): True labels.
    weights (array): Current model weights.
    learning_rate (float): Learning rate for gradient descent.
    
    Returns:
    array: Updated model weights.
    """
    predictions = predict(X, weights)
    errors = y - predictions
    gradient = dot(X.T, errors)
    weights += learning_rate * gradient
    return weights

def train_logistic_regression(X, y, learning_rate, epochs):
    """
    Train the logistic regression model using the provided training data.
    
    Parameters:
    X (array): Input data matrix.
    y (array): True labels.
    learning_rate (float): Learning rate for gradient descent.
    epochs (int): Number of iterations for training.
    
    Returns:
    array: Trained model weights.
    """
    weights = zeros(X[0].length)
    for epoch in range(epochs):
        weights = update_weights(X, y, weights, learning_rate)
    return weights

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of the predictions.
    
    Parameters:
    y_true (array): True labels.
    y_pred (array): Predicted labels.
    
    Returns:
    float: Accuracy score.
    """
    correct = sum(y_true == y_pred)
    return correct / len(y_true)

def precision(y_true, y_pred):
    """
    Calculate the precision of the predictions.
    
    Parameters:
    y_true (array): True labels.
    y_pred (array): Predicted labels.
    
    Returns:
    float: Precision score.
    """
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp)

def recall(y_true, y_pred):
    """
    Calculate the recall of the predictions.
    
    Parameters:
    y_true (array): True labels.
    y_pred (array): Predicted labels.
    
    Returns:
    float: Recall score.
    """
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    """
    Calculate the F1 score of the predictions.
    
    Parameters:
    y_true (array): True labels.
    y_pred (array): Predicted labels.
    
    Returns:
    float: F1 score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r)
