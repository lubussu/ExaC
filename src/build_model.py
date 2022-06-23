import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

path = "../dataset/final_dataset/kepler.csv"
def load_dataset(path):
    return pd.read_csv(path)

dataset = load_dataset(path)
dataset.drop(columns=["pl_name"], inplace=True)

X, y = dataset, dataset.disposition
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print(scores)

#
# dataset = load_dataset(path)
# train_set, test_set = train_test_split(dataset, train_size=0.7, shuffle=True)



