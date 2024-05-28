# Bend Logistic Regression Library

## Overview

This is a data science library written in Bend. It includes functions for training a logistic regression model and evaluating its performance using common metrics such as accuracy, precision, recall, and F1 score.

## Features

- Train logistic regression models
- Evaluate models with accuracy, precision, recall, and F1 score

## Installation

Ensure you have the Bend compiler installed. Clone this repository to your local machine.

## Usage

-Training the Model

To train the logistic regression model, use the train_logistic_regression function. You need to provide the input data X, true labels y, learning rate, and the number of epochs.


import "logistic_regression.bend"

X_train = load_csv('data/X_train.csv')
y_train = load_csv('data/y_train.csv')

learning_rate = 0.01
epochs = 1000

weights = train_logistic_regression(X_train, y_train, learning_rate, epochs)


-Evaluating the Model

To evaluate the model, use the provided metrics functions:

import "logistic_regression.bend"

X_test = load_csv('data/X_test.csv')
y_test = load_csv('data/y_test.csv')

test_predictions = predict(X_test, weights)
y_pred = threshold_predictions(test_predictions)

acc = accuracy(y_test, y_pred)
prec = precision(y_test, y_pred)
rec = recall(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", rec)
print("F1 Score: ", f1)


## Contact

For any questions or inquiries, please contact miguelmmsaraiva1@gmail.com.
