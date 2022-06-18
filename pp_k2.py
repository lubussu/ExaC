import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pre_process():
    k2 = pd.read_csv('./dataset/K2 Objects of Interests.csv', on_bad_lines='skip')
    k2 = k2.dropna(subset=['disposition'])

    print("k2 prev attributes: " + str(k2.columns.size))
    k2.drop(
        columns=['rowid', 'hostname', 'epic_hostname', 'tic_id', 'gaia_id', 'default_flag', 'disp_refname',
                 'discoverymethod', 'disc_year', 'disc_refname', 'disc_pubdate', 'disc_locale', 'disc_facility',
                 'disc_telescope',
                 'disc_instrument', 'soltype', 'pl_controv_flag', 'pl_refname', 'pl_tsystemref', 'st_refname',
                 'sy_refname', 'rowupdate', 'pl_pubdate', 'releasedate', 'pl_nnotes', 'st_nphot', 'st_nrvc',
                 'st_nspec'],
        inplace=True)

    thresh = len(k2) * .5
    k2.dropna(thresh=thresh, axis=1, inplace=True) #remove columns with less than 70% of not nan values

    nunique = k2.nunique() #series with numner of unique value for each column
    cols_to_drop = nunique[nunique == 1].index #indexes of columns with value of nunique == 1
    k2.drop(cols_to_drop, axis=1, inplace=True)

    k2.drop_duplicates(subset='pl_name', keep='first', inplace=True) #remove rows with duplicate value of kepoi_name column

    k2 = k2[k2.columns.drop(list(k2.filter(regex='err')))] #remove columns that contains err in attribute name
    k2 = k2[k2.columns.drop(list(k2.filter(regex='lim')))] #remove columns that contains lim in attribute name

    print("k2 next attributes: " + str(k2.columns.size))

    thresh = k2.columns.size * .8
    k2.dropna(thresh=thresh, axis=0, inplace=True) #remove columns with less than 70% of not nan values

    #remove attributes with a value of corr_m > 0.95 (see on the visual correlation matrix)
    k2.drop(columns=['tran_flag', 'sy_gaiamag', 'sy_jmag', 'sy_hmag', 'sy_kmag',
                     'sy_tmag'])

    k2.to_csv('./dataset/pp_dataset/k2.csv')
    return k2
