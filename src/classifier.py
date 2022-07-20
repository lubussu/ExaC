import pickle
import numpy as np
import pandas as pd


def classify(op):
    to_classify = pd.read_csv('../dataset/final_dataset/to_classify.csv', on_bad_lines='skip')
    to_classify = to_classify[to_classify.columns.drop(list(to_classify.filter(regex='Unnamed')))]
    to_classify.drop(columns=["disposition"], inplace=True)
    to_classify = to_classify.select_dtypes(include=np.number)

    if op == 0:
        clf = pickle.load(open('../documents/obj/classifier_k2-kepler_lc', 'rb'))
    else:
        clf = pickle.load(open('../documents/obj/classifier_k2-kepler', 'rb'))
        to_classify = to_classify[to_classify.columns.drop(list(to_classify.filter(regex='lc_')))]

    predicted = clf.predict(to_classify.values)
    unique, counts = np.unique(predicted, return_counts=True)

    result = np.column_stack((unique, counts))
    print(result)
