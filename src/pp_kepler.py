import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightcurves_features as lcf
from sklearn.impute import KNNImputer


def pre_process():
    kepler = pd.read_csv('../dataset/Kepler Objects of Interests.csv', on_bad_lines='skip', dtype={'kepid': str})
    kepler = kepler.dropna(subset=['koi_disposition'])

    kepler['kepid'] = kepler['kepid'].astype(int).astype(str)

    kepler.drop(
        columns=['rowid', 'kepler_name', 'koi_vet_stat', 'koi_vet_date', 'koi_pdisposition', 'koi_score',
                 'koi_disp_prov', 'koi_comment', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_fittype',
                 'koi_limbdark_mod', 'koi_parm_prov', 'koi_tce_plnt_num', 'koi_tce_delivname',
                 'koi_trans_mod', 'koi_model_dof', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sparprov'],
        inplace=True)

    # thresh = len(kepler) * .5
    # kepler.dropna(thresh=thresh, axis=1, inplace=True)  # remove columns with less than 70% of not nan values

    nunique = kepler.nunique()  # series with number of unique value for each column
    cols_unique_drop = nunique[nunique == 1].index  # indexes of columns with value of nunique == 1
    kepler.drop(cols_unique_drop, axis=1, inplace=True)

    kepler.drop_duplicates(subset='kepoi_name', keep='first')  # remove rows with duplicate value of kepoi_name column

    kepler = kepler[
        kepler.columns.drop(list(kepler.filter(regex='err')))]  # remove columns that contains err in attribute name
    kepler = kepler[
        kepler.columns.drop(list(kepler.filter(regex='lim')))]  # remove columns that contains lim in attribute name

    # thresh = kepler.columns.size * .7
    # kepler.dropna(thresh=thresh, axis=0, inplace=True)  # remove columns with less than 70% of not nan values

    # replace dispostion values in numeric values
    kepler['koi_disposition'].replace(
        {"FALSE POSITIVE": 0, "CONFIRMED": 1, "CANDIDATE": 2},
        inplace=True)

    # feature extraction from lightcurves time series
    kepler = lcf.extract_features(kepler, 'kepid')

    # # handling missing values:
    # kepler2 = kepler.drop(columns=['kepoi_name', 'kepid'])
    #
    # kepler.dropna(axis=1, how='all', inplace=True)
    #
    # k = math.trunc(math.sqrt(len(kepler)))
    # imputer = KNNImputer(n_neighbors=k, weights='uniform', metric='nan_euclidean')
    # kepler_filled = pd.DataFrame(imputer.fit_transform(kepler2), columns=kepler2.columns)
    #
    # kepler_filled.insert(0, 'kepid', kepler['kepid'].values)
    # kepler_filled.insert(0, 'kepoi_name', kepler['kepoi_name'].values)
    # kepler_filled.to_csv('../dataset/pp_dataset/kepler_filled.csv')

    # remove attributes with a value of corr_m > 0.95 (see on the visual correlation matrix)
    kepler.drop(columns=['koi_fwm_sra', 'koi_fwm_sdec', 'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_jmag',
                                'koi_kmag', 'koi_sma', 'koi_ldm_coeff2', 'lc_max', 'lc_min',
                                'lc_meanAbsDev', 'lc_q1'], inplace=True)

    svolume = (4 / 3) * 3.14 * pow(kepler['koi_srad'], 3)
    sdens = kepler['koi_smass'] / svolume
    pdens = sdens * kepler['koi_srho']
    pmass = pdens * (4 / 3) * 3.14 * pow(kepler['koi_prad'], 3)
    kepler['pl_masse'] = pd.Series(pmass.values, index=kepler.index)

    kepler.to_csv('../dataset/pp_dataset/kepler.csv')
    return kepler
