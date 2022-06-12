import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pre_process():
    k2 = pd.read_csv('./dataset/K2 Objects of Interets.csv', on_bad_lines='skip')
    k2 = k2.dropna(subset=['disposition'])
    thresh = len(k2) * .7
    k2.dropna(thresh=thresh, axis=1, inplace=True)
    print(k2['disposition'].unique())
    print(k2.columns.size)
