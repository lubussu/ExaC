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

k2 = pd.read_csv('../dataset/pp_dataset/k2ticids.csv', on_bad_lines='skip')

ids = k2['tic_id']

# for kepler scraping
# ids = ids.map(int)
# ids = ids.map(str)

for x in ids:
    # kepler ids normalization for kepler scraping
    # if len(x) == 6:
    #     target = "000" + x
    #     print(target)
    # if len(x) == 7:
    #     target = "00" + x
    #     print(target)
    # if len(x) == 8:
    #     target = "0" + x
    #     print(target)

    print(x)

    # K2 ids scraping with lightkurve package for K2 scraping
    search_result = lk.search_lightcurve(x, author='K2')

    if not search_result:
        continue

    # general scraping of light curves from MAST archive
    target = search_result[0]

    Obs = Observations.query_criteria(target_name=target.table['target_name'], obs_collection='K2')
    print("******************")

    Prods = Observations.get_product_list(Obs[0])
    print("******************")

    yourProd = Observations.filter_products(Prods, extension='lc.fits',
                                            mrp_only=False)
    print("******************")

    print(yourProd)

    Observations.download_products(yourProd[0], mrp_only=False, cache=False, download_dir='A:/lightcurves')