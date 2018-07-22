# Libraries
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

# Importing dataset
dataset = pd.read_csv("Data.csv")
matrix_of_features = dataset.iloc[:, :-1].values
dependent_variables = dataset.iloc[:, -1].values

# Preparing data - missing data and stuff
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(matrix_of_features[:, 1:3])
matrix_of_features[:, 1:3] = imputer.transform(matrix_of_features[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder1 = LabelEncoder()
matrix_of_features[:, 0] = label_encoder1.fit_transform(matrix_of_features[:, 0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
matrix_of_features = one_hot_encoder.fit_transform(matrix_of_features).toarray()

label_encoder2 = LabelEncoder()
dependent_variables = label_encoder2.fit_transform(dependent_variables)

# Splitting data
from sklearn.model_selection import train_test_split

feat_train, feat_test, dep_train, dep_test = train_test_split(
    matrix_of_features, dependent_variables, test_size=0.2, random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

feat_scaler = StandardScaler()
feat_train = feat_scaler.fit_transform(feat_train)
feat_test = feat_scaler.transform(feat_test)
