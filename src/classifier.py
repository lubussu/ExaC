import pickle
import numpy as np
import pandas as pd


def classify(op):
    to_classify = pd.read_csv('../dataset/final_dataset/to_classify.csv', on_bad_lines='skip')
    to_classify = to_classify[to_classify.columns.drop(list(to_classify.filter(regex='Unnamed')))]
    to_classify.drop(columns=["disposition"], inplace=True)
    to_classify = to_classify.select_dtypes(include=np.number)

    confirmed = pd.read_csv('../dataset/pp_dataset/confirmed.csv', on_bad_lines='skip')

    if op == 0:
        clf = pickle.load(open('../documents/obj/classifier_k2-kepler_lc', 'rb'))
    else:
        clf = pickle.load(open('../documents/obj/classifier_k2-kepler', 'rb'))
        to_classify = to_classify[to_classify.columns.drop(list(to_classify.filter(regex='lc_')))]

    predicted = clf.predict(to_classify.values)
    to_classify["predicted"] = predicted
    indexes = to_classify.index[to_classify["predicted"] == 1]
    confirmed = pd.concat([confirmed, to_classify.iloc[indexes]], ignore_index=True).drop(columns=['predicted'])
    masse = pd.read_csv('/Users/luanabussu/GitHub/ProgettoDM/dataset/K2 Objects of Interests.csv', on_bad_lines='skip')
    masse = masse[['pl_name', 'pl_masse']]
    confirmed = pd.merge(masse, confirmed, on='pl_name')
    print(len(confirmed))
    confirmed.rename(columns={"pl_masse_x": "pl_masse"}, inplace=True)
    confirmed.dropna(subset='pl_masse', inplace=True)
    print(len(confirmed))

    if op == 0:
        confirmed.to_csv("../dataset/final_dataset/confirmed_lc.csv")
    else:
        confirmed.to_csv("../dataset/final_dataset/confirmed.csv")

    unique, counts = np.unique(predicted, return_counts=True)
    result = np.column_stack((unique, counts))
    print(result)
