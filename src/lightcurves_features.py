import math

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style
from astroquery.mast import Mast
import lightkurve as lk
from astroquery.mast import Observations
import pandas as pd

# kepler = pd.read_csv('../dataset/pp_dataset/keplerids.csv', on_bad_lines='skip')
#
# ids = kepler['kepid']
#
# ids = ids.map(int)
# ids = ids.map(str)
# print(len(ids[0]))
#
# for x in ids:
#     if len(x) == 6:
#         target = "000" + x
#         print(target)
#     if len(x) == 7:
#         target = "00" + x
#         print(target)
#     if len(x) == 8:
#         target = "0" + x
#         print(target)

# search_result = lk.search_lightcurve('EPIC 210848071', author='K2')
# print(search_result[0])

    # keplerObs = Observations.query_criteria(target_name='kplr' + target, obs_collection='Kepler')
    # print("******************")
    #
    # keplerProds = Observations.get_product_list(keplerObs[0])
    # print("******************")
    #
    # yourProd = Observations.filter_products(keplerProds, extension='slc.fits',
    #                                         mrp_only=False)
    # print("******************")
    #
    # print(yourProd[0])

    # Observations.download_products(yourProd[0], mrp_only=False, cache=False, download_dir='../lightcurves')

# *************************************************************************************************************

filename = "..\lightcurves\mastDownload\Kepler\kplr000757450_lc_Q011111111111111111/kplr000757450-2009166043257_llc.fits"
fits.info(filename)

with fits.open(filename, mode="denywrite") as hdulist:

    header1 = hdulist[1].header
    binaryext = hdulist[1].data
    # Read in the "BJDREF" which is the time offset of the time array.
    bjdrefi = hdulist[1].header['BJDREFI']
    bjdreff = hdulist[1].header['BJDREFF']

    # Read in the columns of data.
    times = hdulist[1].data['TIME']
    sap_fluxes = hdulist[1].data['SAP_FLUX']
    pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
    timecorr = hdulist[1].data['TIMECORR']

    print(len(times))
    print(len(sap_fluxes))
    print(len(pdcsap_fluxes))
    print(len(timecorr))

print(repr(header1[0:24]))  # repr() prints the info into neat columns
binarytable = Table(binaryext)
print(binarytable[1:5])
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