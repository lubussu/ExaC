import math

import numpy as np
import scipy.stats
import sklearn as sk
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style
from astroquery.mast import Mast
import lightkurve as lk
from astroquery.mast import Observations
import pandas as pd
import os


def extract_features(dataframe, col_name):
    dataframe[['lc_amplitude', 'lc_slope', 'lc_max', 'lc_mean', 'lc_median', 'lc_meanAbsDev', 'lc_min',
               'lc_q1', 'lc_q3', 'lc_q31', 'lc_resBFR', 'lc_skew', 'lc_kurtosis', 'lc_std']] = pd.DataFrame(
        [[np.nan] * 14],
        index=dataframe.index)
    x = 0
    if col_name == 'kepid':
        path = "A:/lightcurves/Kepler"
    if col_name == 'tic_id':
        path = "A:/lightcurves/K2"

    isFile = os.path.isfile(path)
    if not isFile:
        if col_name == 'kepid':
            path = "/Users/luanabussu/lightcurves/Kepler"
        if col_name == 'tic_id':
            path = "/Users/luanabussu/lightcurves/K2"

    for root, dirs, files in os.walk(path):
        for file in files:
            f_path = os.path.join(root, file)
            if file.endswith(".fits"):
                with fits.open(f_path, mode="readonly") as hdulist:
                    # Read in the columns of data.
                    times = hdulist[1].data['TIME']
                    pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

                    times = pd.Series(times).interpolate()

                    lc = pd.Series(pdcsap_fluxes)
                    a = lc.interpolate()

                    col_value = file.removesuffix('.fits')

                    lmax = np.max(a)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_max'] = lmax

                    lmin = np.min(a)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_lmin'] = lmin

                    mean = np.mean(a)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_mean'] = mean

                    q1 = np.quantile(a, 0.25)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_q1'] = q1

                    median = np.quantile(a, 0.5)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_median'] = median

                    q3 = np.quantile(a, 0.75)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_q3'] = q3

                    std = np.std(a)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_std'] = std

                    amp = (lmax - lmin) / 2
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_amplitude'] = amp

                    p = np.polyfit(times, a, deg=1)
                    slope = p[0]
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_slope'] = slope

                    mad_temp = a - median
                    meanAbsDev = np.mean(mad_temp)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_meanAbsDev'] = meanAbsDev

                    q31 = q3 - q1
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_q31'] = q31

                    resBFR = sum(x < mean for x in a) / sum(x > mean for x in a)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_resBFR'] = resBFR

                    skew = scipy.stats.skew(a)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_skew'] = skew

                    kurtosis = scipy.stats.moment(a, moment=4)
                    dataframe.loc[dataframe[col_name] == col_value, 'lc_kurtosis'] = kurtosis

                x = x + 1

    return dataframe

# # # Convert the time array to full BJD by adding the offset back in.
# bjds = times + bjdrefi + bjdreff
#
# plt.figure(figsize=(20, 9))
#
# # # Plot the time, uncorrected and corrected fluxes.
# plt.plot(bjds, sap_fluxes, '-k', label='SAP Flux')
# plt.plot(bjds, pdcsap_fluxes, '-b', label='PDCSAP Flux')
#
# plt.title('dataframe Light Curve')
# plt.legend()
# plt.xlabel('Time (days)')
# plt.ylabel('Flux (electrons/second)')
# plt.show()
