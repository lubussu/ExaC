import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pre_process():
    tess = pd.read_csv('./dataset/TESS Objects of Interests.csv', on_bad_lines='skip')
    tess = tess.dropna(subset=['tfopwg_disp'])
    tess = tess.drop(tess[tess.tfopwg_disp == 'FA'].index)
    tess['tfopwg_disp'].replace(
        {"FP": "FALSE POSITIVE", "KP": "CONFIRMED", "CP": "CONFIRMED", "PC": "CANDIDATE", "APC": "CANDIDATE"},
        inplace=True)

    print("tess prev attributes:" + str(tess.columns.size))

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

    print("tess next attributes:" + str(tess.columns.size))

    thresh = tess.columns.size * .8
    tess.dropna(thresh=thresh, axis=0, inplace=True) #remove columns with less than 70% of not nan values

    #tess.to_csv('./dataset/tess.csv')

    return tess
