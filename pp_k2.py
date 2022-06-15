import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pre_process():
    k2 = pd.read_csv('./dataset/K2 Objects of Interests.csv', on_bad_lines='skip')
    k2 = k2.dropna(subset=['disposition'])

    k2.drop(
        columns=['rowid', 'hostname', 'epic_hostname', 'tic_id', 'gaia_id', 'default_flag', 'disp_refname',
                 'discoverymethod', 'disc_year', 'disc_refname', 'disc_pubdate', 'disc_locale', 'disc_facility',
                 'disc_telescope',
                 'disc_instrument', 'soltype', 'pl_controv_flag', 'pl_refname', 'pl_tsystemref', 'st_refname',
                 'sy_refname',
                 'rastr', 'decstr', 'rowupdate', 'pl_pubdate', 'releasedate', 'pl_nnotes', 'st_nphot', 'st_nrvc',
                 'st_nspec'],
        inplace=True)

    thresh = len(k2) * .7
    k2.dropna(thresh=thresh, axis=1, inplace=True)

    nunique = k2.nunique()
    cols_to_drop = nunique[nunique == 1].index
    k2.drop(cols_to_drop, axis=1, inplace=True)

    k2.drop_duplicates(subset='pl_name', keep='first', inplace=True)

    print(k2['disposition'].unique())
    print(k2.columns.size)

    k2.to_csv('./dataset/k2.csv')
