import glob
import math
import os

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style
from astroquery.mast import Mast
import lightkurve as lk
from astroquery.mast import Observations
import pandas as pd

k2 = pd.read_csv('../dataset/pp_dataset/k2ticids.csv', on_bad_lines='skip')

ids = k2['tic_id']

pathk2 = "A:/lightcurves/K2"
pathkepler = "A:/lightcurves/Kepler"

# # for kepler scraping
# ids = ids.map(int)
# ids = ids.map(str)

for x in ids:
    # # kepler ids normalization for kepler scraping
    # if len(x) == 6:
    #     target = "kplr000" + x
    #     print(target)
    # if len(x) == 7:
    #     target = "kplr00" + x
    #     print(target)
    # if len(x) == 8:
    #     target = "kplr0" + x
    #     print(target)

    print(x)

    # # K2 ids scraping with lightkurve package for K2 scraping
    search_result = lk.search_lightcurve(x, author='K2')

    if not search_result:
        continue

    # # for Kepler
    # target = x
    # for K2
    target = search_result[0]

    # target depends on scraping type
    # general scraping of light curves from MAST archive
    Obs = Observations.query_criteria(target_name=target.table['target_name'], obs_collection='K2')
    print("******************")

    Prods = Observations.get_product_list(Obs[0])
    print("******************")

    yourProd = Observations.filter_products(Prods, extension='lc.fits',
                                            mrp_only=False)
    print("******************")

    print(yourProd)

    # changing path
    download = Observations.download_products(yourProd[0], mrp_only=False, cache=False, download_dir='A:/lightcurves')
    curr_path = download[0].table['Local Path'].value
    os.rename(curr_path[0], os.path.join(pathk2, x + '.fits'))