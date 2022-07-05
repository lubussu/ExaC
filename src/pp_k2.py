import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightcurves_features as lcf
from sklearn.impute import KNNImputer


def pre_process():
    k2 = pd.read_csv('../dataset/K2 Objects of Interests.csv', on_bad_lines='skip', dtype={'tic_id': str})
    k2 = k2.dropna(subset=['disposition'])

    k2['tic_id'] = k2['tic_id'].astype(str)

    k2.drop(
        columns=['rowid', 'hostname', 'epic_hostname', 'gaia_id', 'default_flag', 'disp_refname',
                 'discoverymethod', 'disc_year', 'disc_refname', 'disc_pubdate', 'disc_locale', 'disc_facility',
                 'disc_telescope',
                 'disc_instrument', 'soltype', 'pl_controv_flag', 'pl_refname', 'pl_tsystemref', 'st_refname',
                 'sy_refname', 'rowupdate', 'pl_pubdate', 'releasedate', 'pl_nnotes', 'st_nphot', 'st_nrvc',
                 'st_nspec', 'pl_letter', 'k2_name', 'rastr', 'decstr'],
        inplace=True)

    thresh = len(k2) * .5
    k2.dropna(thresh=thresh, axis=1, inplace=True)  # remove columns with less than 70% of not nan values

    nunique = k2.nunique()  # series with numner of unique value for each column
    cols_to_drop = nunique[nunique == 1].index  # indexes of columns with value of nunique == 1
    k2.drop(cols_to_drop, axis=1, inplace=True)

    k2.drop_duplicates(subset='pl_name', keep='first',
                       inplace=True)  # remove rows with duplicate value of kepoi_name column

    k2 = k2[k2.columns.drop(list(k2.filter(regex='err')))]  # remove columns that contains err in attribute name
    k2 = k2[k2.columns.drop(list(k2.filter(regex='lim')))]  # remove columns that contains lim in attribute name

    thresh = k2.columns.size * .8
    k2.dropna(thresh=thresh, axis=0, inplace=True)  # remove columns with less than 70% of not nan values

    # replace dispostion values in numeric values
    k2['disposition'].replace(
        {"FALSE POSITIVE": 0, "CONFIRMED": 1, "CANDIDATE": 2},
        inplace=True)

    # feature extraction from lightcurves time series
    k2 = lcf.extract_features(k2, 'tic_id')

    # handling missing values:
    k22 = k2.drop(columns=['pl_name', 'pl_ntranspec;;;;', 'tic_id'])

    k22.dropna(axis=1, how='all', inplace=True)

    k = math.trunc(math.sqrt(len(k2)))
    imputer = KNNImputer(n_neighbors=k, weights='uniform', metric='nan_euclidean')
    k2_filled = pd.DataFrame(imputer.fit_transform(k22), columns=k22.columns)

    k2_filled.insert(0, 'tic_id', k2['tic_id'].values)
    k2_filled.insert(0, 'pl_name', k2['pl_name'].values)
    k2_filled.to_csv('../dataset/pp_dataset/k2_filled.csv')

    # remove attributes with a value of corr_m > 0.95 (see on the visual correlation matrix)
    k2_filled.drop(columns=['tran_flag', 'sy_gaiamag', 'sy_jmag', 'sy_hmag', 'sy_kmag',
                            'sy_tmag'], inplace=True)

    k2_filled.to_csv('../dataset/pp_dataset/k2.csv')
    print(k2_filled)
    return k2_filled
