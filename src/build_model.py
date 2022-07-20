import math
import pickle
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, auc, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
import classifier as cf
from util.object_handling import saveObject


def RF_trial(X, y, X_test, y_test):
    max_depths = np.linspace(1, X.shape[1], endpoint=True)
    train_results = []
    test_results = []
    for depth in max_depths:
        rf = RandomForestClassifier(max_depth=int(depth), n_jobs=-1)
        rf.fit(X, y)
        train_pred = rf.predict(X)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    plt.plot(max_depths, train_results, 'b', label='Train AUC')
    plt.plot(max_depths, test_results, 'r', label='Test AUC')
    diff = [y - x for x, y in zip(test_results, train_results)]
    diff = list(filter(lambda d: 0 < d < 0.03, diff))
    depth = int(max_depths[len(diff) - 1])
    plt.vlines(depth, 0, 1)
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.savefig('../documents/images/max_depth.png')
    return depth


def KNN_trial(X, y):
    sqrt = math.trunc(math.sqrt(len(X)))
    error = []
    low = int(sqrt - sqrt / 3)
    up = int(sqrt + sqrt / 3)
    for k in range(low, up):
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, X, y, cv=5)
        error.append(mean_squared_error(y, y_pred))

    plt.plot(range(low, up), error)
    plt.show()
    plt.savefig('../documents/images/best_k.png')
    return error.index(min(error)) + low


def compute_all(data, label, data_t, label_t, balance):
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
            depth = RF_trial(data, label, data_t, label_t)
            clf = RandomForestClassifier(n_estimators=40, max_depth=depth)
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

    print('\n*********** CV Mean Results ***********\n')
    print(df_results.to_markdown())
    print()

    clf = results[scores[scores == max_score].index[0]]
    return clf


def build_model(path):
    file = path.split('/')[-1]
    print("Building model: " + file)
    dataset = pd.read_csv(path, on_bad_lines='skip')
    dataset.drop(dataset.columns[0], axis=1, inplace=True)

    dataset['disposition'] = dataset['disposition'].map(int)

    dataset.drop(columns=["pl_name"], inplace=True)
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Unnamed')))]


    X, y = dataset, dataset.disposition.values

    X.drop(columns=['disposition'], inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = compute_all(X_train, y_train, X_test, y_test, 1)
    clf.fit(X_train, y_train)
    scores = clf.score(X_test, y_test)
    print("Result with test set: ", str(scores))
    print("-------------------------------------------------------------------------------------------")

    pickle.dump(clf, open('../documents/obj/classifier_'+file.split('.')[0], 'wb'))


