import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer


def pre_process():
    tess = pd.read_csv('../dataset/TESS Objects of Interests.csv', on_bad_lines='skip')
    tess = tess.dropna(subset=['tfopwg_disp'])
    tess = tess.drop(tess[tess.tfopwg_disp == 'FA'].index)

    tess.drop(
        columns=['rowid', 'toipfx', 'tid', 'ctoi_alias', 'pl_pnum', 'rastr', 'decstr',
                 'toi_created', 'rowupdate'],
        inplace=True)

    thresh = len(tess) * .5
    tess.dropna(thresh=thresh, axis=1, inplace=True)  # remove columns with less than 70% of not nan values

    nunique = tess.nunique()  # series with numner of unique value for each column
    cols_to_drop = nunique[nunique == 1].index  # indexes of columns with value of nunique == 1
    tess.drop(cols_to_drop, axis=1, inplace=True)

    tess.drop_duplicates(subset='toi', keep='first')  # remove rows with duplicate value of kepoi_name column

    tess = tess[tess.columns.drop(list(tess.filter(regex='err')))]  # remove columns that contains err in attribute name
    tess = tess[tess.columns.drop(list(tess.filter(regex='lim')))]  # remove columns that contains lim in attribute name

    thresh = tess.columns.size * .8
    tess.dropna(thresh=thresh, axis=0, inplace=True) #remove columns with less than 70% of not nan values

    # replace dispostion values in numeric values
    tess['tfopwg_disp'].replace(
        {"FP": 0, "KP": 1, "CP": 1, "PC": 2, "APC": 2},
        inplace=True)

    # handling missing values:
    tess2 = tess.drop(columns=['toi'])

    k = math.trunc(math.sqrt(len(tess)))
    imputer = KNNImputer(n_neighbors=k, weights='uniform', metric='nan_euclidean')
    tess_filled = pd.DataFrame(imputer.fit_transform(tess2), columns=tess2.columns)

    tess_filled.insert(0, 'toi', tess['toi'].values)

    tess_filled.to_csv('../dataset/pp_dataset/tess_filled.csv')
    tess_filled.to_csv('../dataset/pp_dataset/tess.csv')

    return tess_filled
