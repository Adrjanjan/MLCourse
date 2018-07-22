# 1. Data Preprocessing

# A| Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# B| Import Data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,  -1].values

# C| Handling Missing Stuff
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = imputer.fit(X)
X = imputer.transform(X)

# D| Encoding Categorical Data - no use here

# E| Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# F| Feature Scaling - no use here

# --------------------------------- #



# 2. Simple Linear Regression

# A| Fitting Simple Linear Model to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# B| Predicting the Test Set Results
y_pred = regressor.predict(X_test)

# C| Visualising Training Set Results
plt.scatter(X_train, y_train, color='red', edgecolors='black', alpha=0.75)
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# D| Visualising Test Set Results
plt.scatter(X_test, y_test, edgecolors='black', color='green', alpha=0.75)
plt.plot(X_train, regressor.predict(X_train), color='orange')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
