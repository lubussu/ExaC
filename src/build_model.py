import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from collections import Counter

path = "../dataset/final_dataset/k2-kepler.csv"

dataset = pd.read_csv(path)

def exo_random_forest():
    clf = RandomForestClassifier(random_state=5)

    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(scores)

    print("********************************************************")

    clf.fit(X_train, y_train)
    scores2 = clf.score(X_test, y_test)
    print(scores2)


def exo_svm():
    clf = svm.SVC(kernel='linear')

    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(scores)

    print("********************************************************")

    clf.fit(X_train, y_train)
    scores2 = clf.score(X_test, y_test)
    print(scores2)

    clf = svm.SVC(kernel='poly')

    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(scores)

    print("********************************************************")

    clf.fit(X_train, y_train)
    scores2 = clf.score(X_test, y_test)
    print(scores2)


def exo_knn():
    clf = KNeighborsClassifier(n_neighbors=math.trunc(math.sqrt(len(dataset))))

    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(scores)

    print("********************************************************")

    clf.fit(X_train, y_train)
    scores2 = clf.score(X_test, y_test)
    print(scores2)


def exo_log_reg():
    clf = LogisticRegression()

    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(scores)

    print("********************************************************")

    clf.fit(X_train, y_train)
    scores2 = clf.score(X_test, y_test)
    print(scores2)


def exo_NB():
    clf = GaussianNB()

    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(scores)

    print("********************************************************")

    clf.fit(X_train, y_train)
    scores2 = clf.score(X_test, y_test)
    print(scores2)


path = "../dataset/final_dataset/k2-kepler.csv"

dataset = pd.read_csv(path)
dataset.drop(dataset.columns[0], axis=1, inplace=True)

print(dataset.to_markdown())

dataset.drop(columns=["pl_name"], inplace=True)

X, y = dataset, dataset.disposition.values

X.drop(columns=['disposition'], inplace=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

counter = Counter(y_train)

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
counter = Counter(y_train)

exo_random_forest()