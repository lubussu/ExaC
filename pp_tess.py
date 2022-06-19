import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer


def pre_process():
    tess = pd.read_csv('./dataset/TESS Objects of Interests.csv', on_bad_lines='skip')
    tess = tess.dropna(subset=['tfopwg_disp'])
    tess = tess.drop(tess[tess.tfopwg_disp == 'FA'].index)
    tess['tfopwg_disp'].replace(
        {"FP": -1, "KP": 1, "CP": 1, "PC": 0, "APC": 0},
        inplace=True)

    print("tess prev attributes:" + str(tess.columns.size))
    print("tess prev rows: " + str(len(tess)))

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

    tess.to_csv('./dataset/pp_dataset/tess.csv')

    print("tess next attributes:" + str(tess.columns.size))

    tess2 = tess.drop(columns=['toi', 'tfopwg_disp'])

    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    tess_filled = pd.DataFrame(imputer.fit_transform(tess2), columns=tess2.columns)

    print(tess_filled.isnull().sum())

    name_col = tess['toi']
    disposition_col = tess['tfopwg_disp']
    tess_filled = tess_filled.join(name_col)
    tess_filled = tess_filled.join(disposition_col)

    tess_filled.to_csv('./dataset/pp_dataset/tess_filled.csv')
    return tess
