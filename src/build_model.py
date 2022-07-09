import math
from os import truncate

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind


def compute_all(data_in, label_in, balance):
    clf_list = ['KNN', 'Gaussian NB', 'Bernoulli NB', 'Logistic Regression', 'Decision Tree', 'Random Forest',
                'Linear SVM', 'Poly SVM', 'RBF SVM']
    data = data_in
    label = label_in

    results = {'KNN': '',
               'Gaussian NB': '',
               'Bernoulli NB': '',
               'Logistic Regression': '',
               'Decision Tree': '',
               'Random Forest': '',
               'Linear SVM': '',
               'Poly SVM': '',
               'RBF SVM': ''}

    df_results = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC'])

    ttest_acc = {'KNN': '',
                 'Gaussian NB': '',
                 'Bernoulli NB': '',
                 'Logistic Regression': '',
                 'Decision Tree': '',
                 'Random Forest': '',
                 'Linear SVM': '',
                 'Poly SVM': '',
                 'RBF SVM': ''}

    df_accs = pd.DataFrame(ttest_acc, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    for s in clf_list:
        if s == 'KNN':
            clf = KNeighborsClassifier()
        elif s == 'Gaussian NB':
            clf = GaussianNB()
        elif s == 'Bernoulli NB':
            clf = BernoulliNB()
        elif s == 'Logistic Regression':
            clf = LogisticRegression(max_iter=200)
        elif s == 'Decision Tree':
            clf = DecisionTreeClassifier()
        elif s == 'Random Forest':
            clf = RandomForestClassifier()
        elif s == 'Linear SVM':
            clf = svm.SVC(kernel='linear')
        elif s == 'Poly SVM':
            clf = svm.SVC(kernel='poly')
        elif s == 'RBF SVM':
            clf = svm.SVC()
        else:
            print("Error in reading the classifiers!")
            break

        if balance == 1:
            oversample = SMOTE()
            data, label = oversample.fit_resample(data, label)

        accuracy = cross_val_score(clf, data, label, cv=10)
        precision = cross_val_score(clf, data, label, cv=10, scoring='precision')
        recall = cross_val_score(clf, data, label, cv=10, scoring='recall')
        f1score = cross_val_score(clf, data, label, cv=10, scoring='f1')
        roc_auc = cross_val_score(clf, data, label, cv=10, scoring='roc_auc')

        df_results.loc['Accuracy', s] = '%.3f' % np.mean(accuracy)
        df_results.loc['Precision', s] = '%.3f' % np.mean(precision)
        df_results.loc['Recall', s] = '%.3f' % np.mean(recall)
        df_results.loc['F1-score', s] = '%.3f' % np.mean(f1score)
        df_results.loc['ROC AUC', s] = '%.3f' % np.mean(roc_auc)

        df_accs[s] = accuracy

        print(s + ' done!')

    print(df_results.to_markdown())

    print('\nCV accuracy scores:')
    print(df_accs.to_markdown())

    print('\nT-test matrix (scores/p-values):')
    df_ttest = ttest_matrix(df_accs)

    return df_results, df_ttest


def ttest_matrix(ttest_accs):

    ttest_dct = {'KNN': '',
                 'Gaussian NB': '',
                 'Bernoulli NB': '',
                 'Logistic Regression': '',
                 'Decision Tree': '',
                 'Random Forest': '',
                 'Linear SVM': '',
                 'Poly SVM': '',
                 'RBF SVM': ''}

    ttest_results = pd.DataFrame(ttest_dct, index=['KNN', 'Gaussian NB', 'Bernoulli NB', 'Logistic Regression',
                                                   'Decision Tree', 'Random Forest', 'Linear SVM', 'Poly SVM',
                                                   'RBF SVM'])

    for c in ttest_accs:
        for r in ttest_accs:
            if r == c:
                ttest_results.loc[r, c] = ' '
                continue
            s, p = ttest_ind(ttest_accs[c], ttest_accs[r])
            ttest_results.loc[r, c] = '%.3f / %.3f' % (s, p)

    print(ttest_results.to_markdown())

    return ttest_results


path = "../dataset/final_dataset/k2-kepler.csv"

dataset = pd.read_csv(path)
dataset.drop(dataset.columns[0], axis=1, inplace=True)

dataset['disposition'] = dataset['disposition'].map(int)

dataset.drop(columns=["pl_name"], inplace=True)

X, y = dataset, dataset.disposition.values

X.drop(columns=['disposition'], inplace=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

compute_all(X_train, y_train, 0)
