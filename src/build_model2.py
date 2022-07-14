import math
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
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
from termcolor import colored
from sklearn.model_selection import cross_val_predict

def KNN_trial(X, y):
    sqrt = math.trunc(math.sqrt(len(X)))
    error = []
    low = int(sqrt - sqrt / 2)
    up = int(sqrt + sqrt / 2)
    for k in range(low, up):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, X, y, cv=5)
        error.append(mean_squared_error(y, y_pred))

    plt.plot(range(low, up), error)
    plt.show()
    # plt.savefig('documents/images/best_k.png')
    return error.index(min(error)) + low


def compute_all(data, label, balance):
    clf_list = ['KNN', 'Gaussian NB', 'Logistic Regression', 'Random Forest', 'RBF SVM']

    results = {'KNN': '',
               'Gaussian NB': '',
               'Logistic Regression': '',
               'Random Forest': '',
               'RBF SVM': ''}

    df_scores = pd.DataFrame(results, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    df_results = pd.DataFrame(results,
                              index=['Accuracy', 'Acc Std', 'Precision', 'Recall', 'F1-score', 'ROC AUC', 'Time (s)'])

    if balance == 1:
        oversample = SMOTE()
        data, label = oversample.fit_resample(data, label)

    for s in clf_list:
        if s == 'KNN':
            clf = KNeighborsClassifier(KNN_trial(data, label))
        elif s == 'Gaussian NB':
            clf = GaussianNB()
        elif s == 'Logistic Regression':
            clf = LogisticRegression(max_iter=200)
        elif s == 'Random Forest':
            clf = RandomForestClassifier()
        elif s == 'RBF SVM':
            clf = svm.SVC()
        else:
            print("Error in reading the classifiers!")
            break

        results[s] = clf

        begin = time.time()
        scores = cross_validate(clf, data, label, cv=10, scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))
        end = time.time()
        ex_time = end - begin

        df_results.loc['Accuracy', s] = '%.3f' % np.mean(scores['test_accuracy'])
        df_results.loc['Acc Std', s] = '%.3f' % np.std(scores['test_accuracy'])
        df_results.loc['Precision', s] = '%.3f' % np.mean(scores['test_precision'])
        df_results.loc['Recall', s] = '%.3f' % np.mean(scores['test_recall'])
        df_results.loc['F1-score', s] = '%.3f' % np.mean(scores['test_f1'])
        df_results.loc['ROC AUC', s] = '%.3f' % np.mean(scores['test_roc_auc'])
        df_results.loc['Time (s)', s] = '%.3f' % ex_time

        if balance == 1:
            df_scores[s] = scores['test_f1']
        else:
            df_scores[s] = scores['test_accuracy']

    if balance == 1:
        max_score = max(df_results.loc['F1-score'])
        scores = df_results.loc['F1-score']
        print('\n*********** CV F1 Scores ***********\n')
    else:
        max_score = max(df_results.loc['Accuracy'])
        scores = df_results.loc['Accuracy']
        print('\n*********** CV F1 Accuracy ***********\n')
    print(df_scores.to_markdown())

    print('\n*********** CV Results ***********\n')
    print(df_results.to_markdown())
    print()

    clf = results[scores[scores == max_score].index[0]]
    return df_results, clf


path = "../dataset/final_dataset/k2-kepler.csv"

dataset = pd.read_csv(path, on_bad_lines='skip')
dataset.drop(dataset.columns[0], axis=1, inplace=True)

values = dataset['disposition'].value_counts() / len(dataset)
majority = values[values.idxmax()]

if majority > 0.6:
    balance = 1
else:
    balance = 0

dataset['disposition'] = dataset['disposition'].map(int)

dataset.drop(columns=["pl_name"], inplace=True)

X, y = dataset, dataset.disposition.values

X.drop(columns=['disposition'], inplace=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

results, clf = compute_all(X_train, y_train, balance)
clf.fit(X_train, y_train)
scores = clf.score(X_test, y_test)
print(scores)

#Feature importance
if isinstance(clf, RandomForestClassifier):
    importances = clf.feature_importances_
    forest_importances = pd.Series(importances, index=dataset.columns).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize = (40, 34))
    forest_importances.plot.barh()
    ax.set_title("Feature importances Random Forest", fontsize = 40)
    ax.tick_params(axis='y',labelsize=25)