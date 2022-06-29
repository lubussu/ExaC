import math

import numpy as np
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

filename = "A:/lightcurves/mastDownload/Kepler/kplr000757450_lc_Q011111111111111111/kplr000757450-2009166043257_llc.fits"

isFile = os.path.isfile(filename)

if not isFile:
    filename = "/Users/luanabussu/Kepler/kplr000757450_lc_Q011111111111111111/kplr000757450-2009166043257_llc.fits"

fits.info(filename)

with fits.open(filename, mode="readonly") as hdulist:
    header1 = hdulist[1].header
    binaryext = hdulist[1].data
    # Read in the "BJDREF" which is the time offset of the time array.
    bjdrefi = hdulist[1].header['BJDREFI']
    bjdreff = hdulist[1].header['BJDREFF']

    # Read in the columns of data.
    times = hdulist[1].data['TIME']
    sap_fluxes = hdulist[1].data['SAP_FLUX']
    pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

    a = pd.Series(pdcsap_fluxes)
    a.interpolate(inplace=True)

    # print(len(times))
    # print(len(sap_fluxes))
    print(len(pdcsap_fluxes))
    print(np.isnan(a).sum())
    print(np.max(a))
    print(np.min(a))
    print(np.mean(a))
    print(np.quantile(a, 0.25))
    print(np.quantile(a, 0.5))
    print(np.quantile(a, 0.75))
    print(np.std(a))

# print(repr(header1[0:24]))  # repr() prints the info into neat columns
# binarytable = Table(binaryext)
# print(binarytable.info)

# x = 0
#
# for root, dirs, files in os.walk("A:/lightcurves"):
#     for file in files:
#         if file.endswith(".fits"):
#             print(file)
#             x = x + 1
#
# print(x)

# # Convert the time array to full BJD by adding the offset back in.
bjds = times + bjdrefi + bjdreff

plt.figure(figsize=(20, 9))

# # Plot the time, uncorrected and corrected fluxes.
plt.plot(bjds, sap_fluxes, '-k', label='SAP Flux')
plt.plot(bjds, pdcsap_fluxes, '-b', label='PDCSAP Flux')

plt.title('Kepler Light Curve')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Flux (electrons/second)')
plt.show()


