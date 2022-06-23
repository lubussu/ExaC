import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pp_tess
import pp_k2
import pp_kepler


def data_integration():
    k2 = pp_k2.pre_process()
    kepler = pp_kepler.pre_process()
    tess = pp_tess.pre_process()

    kepler.rename(columns={"kepoi_name": "pl_name",
                           "koi_disposition": "disposition",
                           "koi_period": "pl_orbper",
                           "koi_prad": "pl_rade",
                           "koi_eccen": "pl_orbeccen",
                           "koi_sma": "pl_orbsmax",
                           "koi_srho": "pl_dens",
                           "koi_eccen": "pl_orbeccen",
                           "koi_insol": "pl_insol",
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
                           "koi_kepmag": "sy_kepmag"}, inplace=True)

    tess.rename(columns={"toi": "pl_name", "tfopwg_disp": "disposition"}, inplace=True)

    common_cols = list(set.intersection(set(k2), set(kepler)))
    k2 = k2[k2.columns.intersection(common_cols)]
    kepler = kepler[kepler.columns.intersection(common_cols)]
    dataframe = pd.concat([k2, kepler], ignore_index=True)
    dataframe.drop(dataframe.index[dataframe['disposition'] == 0], inplace=True)
    dataframe.to_csv('../dataset/final_dataset/k2-kepler.csv')

    common_cols = list(set.intersection(set(k2), set(kepler), set(tess)))
    k2 = k2[k2.columns.intersection(common_cols)]
    kepler = kepler[kepler.columns.intersection(common_cols)]
    tess = tess[tess.columns.intersection(common_cols)]
    dataframe = pd.concat([k2, kepler, tess], ignore_index=True)
    dataframe.drop(dataframe.index[dataframe['disposition'] == 0], inplace=True)
    dataframe.to_csv('../dataset/final_dataset/all.csv')

    kepler.drop(kepler.index[kepler['disposition'] == 0], inplace=True)
    kepler.to_csv('../dataset/final_dataset/kepler.csv')

    k2.drop(kepler.index[kepler['disposition'] == 0], inplace=True)
    k2.to_csv('../dataset/final_dataset/k2.csv')

    tess.drop(kepler.index[kepler['disposition'] == 0], inplace=True)
    tess.to_csv('../dataset/final_dataset/tess.csv')

