"""
importing necessary libraries
"""
import pandas as pd
import numpy as np
import argparse
import pickle 
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


"""
reading and splitting the data
"""
df = pd.read_csv('../ml/mlbookcamp-code/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.churn = (df.churn == 'Yes').astype(int)


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']



def train(df, y):
    """
    This function accepts a feature matrix and a target,
    and trains a logistic regression model on it
    params: feature matrix, target series
    returns: instances of DictVectorizer and trained model
    rtype: object
    """
    features = ['tenure', 'monthlycharges', 'contract']
    dicts = df[features].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)
    model = LogisticRegression().fit(X, y)
    return dv, model


def predict(df, dv, model):
    """
    This function takes creates predictions for test data
    params: instances of DictVectorizer and trained model, test dataframe
    returns: predictions
    rtype: array
    """
    features = ['tenure', 'monthlycharges', 'contract']
    dicts = df[features].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred
    

"""
Training and testing model
"""
dv, model = train(df_train_full, df_train_full.churn.values)
y_pred = predict(df_test, dv, model)

"""
Saving objects used for predictions to a user specified file
The file name is specified at runtime
"""
try:
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, help="Where the model is saved to" )

    args = parser.parse_args()


    with open(args.name, 'wb') as output:
        pickle.dump((dv, model), output)

    print(f"model is saved to {args.name} ")
except Exception as error:
    print(error)