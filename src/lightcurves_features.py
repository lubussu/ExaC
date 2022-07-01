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

kepler = pd.read_csv('../dataset/Kepler Objects of Interests.csv', on_bad_lines='skip')

kepler[['lc_amplitude', 'lc_slope', 'lc_max', 'lc_mean', 'lc_median', 'lc_meanAbsDev', 'lc_min',
        'lc_q1', 'lc_q31', 'lc_resBFR', 'lc_skew', 'lc_kurtosis', 'lc_std']] = pd.DataFrame([[np.nan] * 13], index=kepler.index)

path = "A:/lightcurves/mastDownload/Kepler"
isFile = os.path.isfile(path)
if not isFile:
    path = "/Users/luanabussu/Kepler"

x = 0

for root, dirs, files in os.walk(path):
    for file in files:
        f_path = os.path.join(root, file)
        if file.endswith(".fits"):
            with fits.open(f_path, mode="readonly") as hdulist:
                header1 = hdulist[1].header
                binaryext = hdulist[1].data
                # Read in the "BJDREF" which is the time offset of the time array.
                bjdrefi = hdulist[1].header['BJDREFI']
                bjdreff = hdulist[1].header['BJDREFF']

                # Read in the columns of data.
                times = hdulist[1].data['TIME']
                sap_fluxes = hdulist[1].data['SAP_FLUX']
                pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

                times = pd.Series(times).interpolate()

                lc = pd.Series(pdcsap_fluxes)
                a = lc.interpolate()

                k_id = file.replace("kplr000", "").replace("kplr00", "").replace("kplr0", "")
                k_id = k_id.split("-")[0]

                lmax = np.max(a)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_max'] = lmax

                lmin = np.min(a)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_lmin'] = lmin

                mean = np.mean(a)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_mean'] = mean

                q1 = np.quantile(a, 0.25)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_q1'] = q1

                median = np.quantile(a, 0.5)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_median'] = median

                q3 = np.quantile(a, 0.75)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_q3'] = q3

                std = np.std(a)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_std'] = std

                amp = (lmax - lmin) / 2
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_amplitude'] = amp

                p = np.polyfit(times, a, deg=1)
                slope = p[0]
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_slope'] = slope

                mad_temp = a - median
                meanAbsDev = np.mean(mad_temp)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_meanAbsDev'] = meanAbsDev

                q31 = q3 - q1
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_q31'] = q31

                resBFR = sum(x < mean for x in a) / sum(x > mean for x in a)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_resBFR'] = resBFR

                skew = scipy.stats.skew(a)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_skew'] = skew

                kurtosis = scipy.stats.moment(a, moment=4)
                kepler.loc[kepler['kepid'] == int(k_id), 'lc_kurtosis'] = kurtosis

            x = x + 1

print(kepler)

# # # Convert the time array to full BJD by adding the offset back in.
# bjds = times + bjdrefi + bjdreff
#
# plt.figure(figsize=(20, 9))
#
# # # Plot the time, uncorrected and corrected fluxes.
# plt.plot(bjds, sap_fluxes, '-k', label='SAP Flux')
# plt.plot(bjds, pdcsap_fluxes, '-b', label='PDCSAP Flux')
#
# plt.title('Kepler Light Curve')
# plt.legend()
# plt.xlabel('Time (days)')
# plt.ylabel('Flux (electrons/second)')
# plt.show()


