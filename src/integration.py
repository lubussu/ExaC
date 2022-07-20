import math
import os
import pandas as pd
from sklearn.impute import KNNImputer
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
    path = "../dataset/pp_dataset/"
    isK2 = os.path.isfile(path+"k2.csv")
    isKepler = os.path.isfile(path + "kepler.csv")
    if not isK2:
        k2 = pp_k2.pre_process()
    else:
        k2 = pd.read_csv(path+"k2.csv", on_bad_lines='skip')
    if not isKepler:
        kepler = pp_kepler.pre_process()
    else:
        kepler = pd.read_csv(path+"kepler.csv", on_bad_lines='skip')

    kepler = kepler_rename(kepler)

    common_cols = list(set.intersection(set(k2), set(kepler)))
    k2 = k2[k2.columns.intersection(common_cols)]
    kepler = kepler[kepler.columns.intersection(common_cols)]
    dataframe = pd.concat([k2, kepler], ignore_index=True)

    thresh = dataframe.columns.size * .7
    dataframe.dropna(thresh=thresh, axis=0, inplace=True)  # remove rows with less than 70% of not nan values

    thresh = len(dataframe) * .5
    dataframe.dropna(thresh=thresh, axis=1, inplace=True)  # remove columns with less than 50% of not nan values

    #dataframe.dropna(axis=1, how='all', inplace=True)

    dataframe2 = dataframe.drop(columns=['pl_name', 'disposition'])

    k = math.trunc(math.sqrt(len(dataframe)))
    imputer = KNNImputer(n_neighbors=k, weights='uniform', metric='nan_euclidean')
    dataframe_filled = pd.DataFrame(imputer.fit_transform(dataframe2), columns=dataframe2.columns)

    dataframe_filled.insert(0, 'disposition', dataframe['disposition'].values)
    dataframe_filled.insert(0, 'pl_name', dataframe['pl_name'].values)

    to_classify = dataframe_filled.loc[dataframe_filled["disposition"] == 2]
    dataframe_filled.drop(dataframe_filled.index[dataframe_filled['disposition'] == 2], inplace=True)
    to_classify.to_csv('../dataset/final_dataset/to_classify.csv')
    confirmed = dataframe_filled.loc[dataframe_filled["disposition"] == 1]
    confirmed.to_csv('../dataset/pp_dataset/confirmed.csv')

    dataframe_filled.to_csv('../dataset/final_dataset/k2-kepler_lc.csv')
    dataframe_filled = dataframe_filled[dataframe_filled.columns.drop(list(dataframe_filled.filter(regex='lc_')))]
    dataframe_filled.to_csv('../dataset/final_dataset/k2-kepler.csv')

