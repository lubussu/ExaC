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

    thresh = len(tess) * .7
    tess.dropna(thresh=thresh, axis=1, inplace=True)

    nunique = tess.nunique()
    cols_to_drop = nunique[nunique == 1].index
    tess.drop(cols_to_drop, axis=1, inplace=True)

    tess.drop_duplicates(subset='toi', keep='first')

    print(tess['tfopwg_disp'].unique())
    print(tess.columns.size)

    tess.to_csv('./dataset/tess.csv')
