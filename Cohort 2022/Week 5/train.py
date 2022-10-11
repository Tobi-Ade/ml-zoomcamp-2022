"""
importing necessary libraries 
"""

#Analysis packages
from distutils.log import error
import pandas as pd
import numpy as np

#Visualization packages
import seaborn as sns
import matplotlib.pyplot as plt

#machine learning packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle
import warnings
warnings.filterwarnings('ignore')


"""
reading the data
"""
print("Parsing the data...")
try:
    df = pd.read_csv('train.csv')
    print("Parse succesful!")
except error as e:
    print(e)


"""
Splitting the data 
"""
print("Performing data preparation...")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_full_train = df_full_train.price_range.values
y_train = df_train.price_range.values
y_val = df_val.price_range.values
y_test = df_test.price_range.values

del df_full_train['price_range']
del df_train['price_range']
del df_val['price_range']
del df_test['price_range']

def train(df, y):
    """
    This function takes in a set of features and targets, 
    and fits them on a logistic regression model
    params: features, target
    returns: standard scaler, model objects
    rtype: object
    """
    sc = StandardScaler()
    X = sc.fit_transform(df)

    model = LogisticRegression(random_state=1)
    model.fit(X, y)
    
    return sc, model

def predict(df, sc, model):
    """
    This function predicts a target class for a set of features
    params: features, standard scaler and model objects
    returns: target class
    rtype: integer
    """
    X = sc.transform(df)
    y_pred = model.predict(X)

    return y_pred

print("Training and validating Logistic Regression model...")

sc, model = train(df_full_train, y_full_train)
print("Training successful!")

y_pred = predict(df_test, sc, model)

score = accuracy_score(y_test, y_pred)

print("Validation successful!")


"""
Saving the model
"""
print("Saving the model...")
output_file = "pred_file.bin"


with open(output_file, 'wb') as out_file:
    pickle.dump((sc, model), out_file)

print(f"model successfully saved to {output_file}")
