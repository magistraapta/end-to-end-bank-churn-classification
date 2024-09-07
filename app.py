from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import mlflow

# setup mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow Quickstart")

# load dataset
df = pd.read_csv('data/bank-churn.csv')

# preprocessing data
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

X = df[['Age', 'CreditScore', 'Balance', 'EstimatedSalary']]
y = df['Exited']

ros = RandomOverSampler()
X, y = ros.fit_resample(X,y)

encoder = LabelEncoder()

X['Geography'] = encoder.fit_transform(X['Geography'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# run mlflow tracking
with mlflow.start_run():

    # initialize model
    model = RandomForestClassifier()

    # train model
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    
    # evaluate model using this metrics
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"recall: {recall}")
    print(f"f1: {f1}")
    print(f"precision: {precision}")
    print(f"accuracy: {accuracy}")
    
    # log the model and metrics into mlflow
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_metric("recall score", recall)
    mlflow.log_metric("f1 score", f1)
    mlflow.log_metric("precision score", precision)
    mlflow.log_metric("accuracy score", accuracy)