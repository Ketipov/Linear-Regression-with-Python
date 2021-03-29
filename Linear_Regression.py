#!/usr/bin/env python
# coding: utf-8
# Linear Regression - example Agreeableness (independent_var) and Product descriptions (dependent_var)
# @Rumen Ketipov


# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Importing the dataset
df = pd.read_csv(r"your directory")
df.describe()


# Assigning the input and output values with other words dividing data into attributes and labels
X = df.iloc[0:, 35].values.reshape(-1,1) # Agreeableness (indep_v)
y = df.iloc[0:, 1].values.reshape(-1,1)  # Product descriptions (dep_v)


# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# stratify=y - to reserve the proportion of target as in original dataset, 
        # in the train and test datasets as well especially useful when working around classification problems 


# Import the regressor
from sklearn.linear_model import LinearRegression

# Create the regressor object
regressor = LinearRegression()

# Training the algorithm
regressor.fit(X_train, y_train) 
                                
# To retrieve the intercept
print(regressor.intercept_)

# For retrieving the slope
print(regressor.coef_)

# Make prediction
y_pred = regressor.predict(X_test)
y_pred # List predicted values


# To display actual and predicted values
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

# Bar chart - actual and predicted values
df1 = df.head(60)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# Correlation with regression
sns.pairplot(df, kind="reg")
# Density
sns.pairplot(df, diag_kind="kde")
plt.show()


# Plotting again for better visualization
plt.xlabel('Agreeableness', fontsize=10)
plt.ylabel('Product description', fontsize=10)
X = list(range(len(y)))
plt.scatter(X_train, y_train, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Predicted")
plt.legend()
plt.show()


# Calculation of evaluation metrics
# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
import sklearn.metrics as metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_test - y_pred)
mape = 100 * (errors / y_test)

# Calculate and display accuracy 
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Calculate and print MAPE again
def MAPE(y_pred, y_test):
    return ( abs((y_test - y_pred) / y_test).mean()) * 100
print ('My MAPE: ' + str(MAPE(y_pred, y_test)) + ' %' )


# Plotting of predicted values
  # plt.xlabel('Agreeableness', fontsize=14)
  # plt.ylabel('Product description', fontsize=14)

# Plotting of prediction
plt.plot(y_pred, label='Prediction')
plt.show()


# Plotting of predicted and actual values for better understanding
plt.plot(y_test, label='Actual values')
plt.plot(y_pred, color="red", label='Predicted values')
plt.legend()
plt.show()


# Jointplot - to visualize how data is distributed
sns.jointplot(x=y_test, y=y_pred, kind='kde')
#plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.show()


# Boxplot - actual and predicted values - how does the algorithm manage outliers
import seaborn as sns
sns.boxplot(x=y_test.flatten(), y=y_pred.flatten()) 
plt.ylabel('Predicted', fontsize=14)
plt.xlabel('Actual', fontsize=14)
plt.show()

