# Task-6

//First, we need to import the necessary libraries such as numpy, pandas, and sklearn.

#python code#

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

//Next, we need to load our dataset. For this example, we'll use the Boston Housing dataset, which is available in scikit-learn.

#python code#
from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['PRICE'] = boston_dataset.target

//Now, we'll select the features we want to use as input for our machine learning model, which are the "RM" feature (average number of rooms per dwelling) and the "LSTAT" feature (percentage of lower status of the population).

#python code#
X = pd.DataFrame(np.c_[boston['RM'], boston['LSTAT']], columns=['RM', 'LSTAT'])
y = boston['PRICE']

//We'll then split our dataset into training and testing sets.

#python code#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

Next, we'll create a Linear Regression model and train it on our training data.

#python code#
model = LinearRegression()
model.fit(X_train, y_train)

//Finally, we'll use our trained model to make predictions on our test data and evaluate the model's performance using mean squared error.

#python code#
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

//That's it! This is a basic example of a machine learning program to predict the price of a house based on its size and number of bedrooms. You can try different algorithms and hyperparameters to improve the model's performance.
