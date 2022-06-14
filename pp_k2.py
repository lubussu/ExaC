import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pre_process():
    k2 = pd.read_csv('./dataset/K2 Objects of Interests.csv', on_bad_lines='skip')
    k2 = k2.dropna(subset=['disposition'])

    thresh = len(k2) * .7
    k2.dropna(thresh=thresh, axis=1, inplace=True)

    nunique = k2.nunique()
    cols_to_drop = nunique[nunique == 1].index
    k2.drop(cols_to_drop, axis=1, inplace=True)

    k2.drop_duplicates(subset=['pl_name'], keep='last')

    print(k2['disposition'].unique())
    print(k2.columns.size)

    # k2.to_csv('./dataset/k2.csv')
