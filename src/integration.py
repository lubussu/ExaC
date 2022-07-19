import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

import pp_tess
import pp_k2
import pp_kepler


def kepler_rename(kepler):
    kepler.rename(columns={"kepoi_name": "pl_name",
                           "koi_disposition": "disposition",
                           "koi_period": "pl_orbper",
                           "koi_prad": "pl_rade",
                           "koi_sma": "pl_orbsmax",
                           "koi_eccen": "pl_orbeccen",
                           "koi_teq": "pl_eqt",
                           "koi_incl": "pl_orbincl",
                           "koi_longp": "pl_orblper",
                           "koi_impact": "pl_imppar",
                           "koi_duration": "pl_trandur",
                           "koi_depth": "pl_trandep",
                           "koi_ror": "pl_ratror",
                           "koi_dor": "pl_ratdor",
                           "koi_steff": "st_teff",
                           "koi_slogg": "st_logg",
                           "koi_smet": "st_met",
                           "koi_srad": "st_rad",
                           "koi_smass": "st_mass",
                           "koi_sage": "st_age",
                           "koi_kepmag": "sy_kepmag",
                           "koi_count": "sy_pnum"}, inplace=True)
    return kepler


def data_integration():
    kepler = pp_kepler.pre_process()
    k2 = pp_k2.pre_process()

    kepler = kepler_rename(kepler)

    kepler.drop(kepler.index[kepler['disposition'] == 2], inplace=True)
    kepler.to_csv('../dataset/final_dataset/kepler.csv')

    k2.drop(k2.index[k2['disposition'] == 2], inplace=True)
    k2.to_csv('../dataset/final_dataset/k2.csv')

    common_cols = list(set.intersection(set(k2), set(kepler)))
    k2 = k2[k2.columns.intersection(common_cols)]
    kepler = kepler[kepler.columns.intersection(common_cols)]
    dataframe = pd.concat([k2, kepler], ignore_index=True)
    dataframe.drop(dataframe.index[dataframe['disposition'] == 2], inplace=True)

    thresh = len(dataframe) * .5
    dataframe.dropna(thresh=thresh, axis=1, inplace=True)  # remove columns with less than 70% of not nan values

    thresh = dataframe.columns.size * .7
    dataframe.dropna(thresh=thresh, axis=0, inplace=True)  # remove columns with less than 70% of not nan values

    dataframe.dropna(axis=1, how='all', inplace=True)

    dataframe2 = dataframe.drop(columns=['pl_name', 'disposition'])

    k = math.trunc(math.sqrt(len(dataframe)))
    imputer = KNNImputer(n_neighbors=k, weights='uniform', metric='nan_euclidean')
    dataframe_filled = pd.DataFrame(imputer.fit_transform(dataframe2), columns=dataframe2.columns)

    dataframe_filled.insert(0, 'disposition', dataframe['disposition'].values)
    dataframe_filled.insert(0, 'pl_name', dataframe['pl_name'].values)

    dataframe_filled.to_csv('../dataset/final_dataset/k2-kepler_lc.csv')
    dataframe_filled = dataframe_filled[dataframe_filled.columns.drop(list(dataframe_filled.filter(regex='lc_')))]
    dataframe_filled.to_csv('../dataset/final_dataset/k2-kepler.csv')
