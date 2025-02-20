import os
from astroquery.mast import Observations
import pandas as pd

df = pd.read_csv('../dataset/pp_dataset/keplerids.csv', on_bad_lines='skip')

ids = df['kepid'].drop_duplicates()

pathk2 = "A:/lightcurves/K2"
pathkepler = "A:/lightcurves/Kepler"
pathtess = "A:/lightcurves/TESS"

# for kepler scraping
ids = ids.map(int)
ids = ids.map(str)

for x in ids:
    # kepler ids normalization for kepler scraping
    if len(x) == 6:
        target = "kplr000" + x
        print(target)
    if len(x) == 7:
        target = "kplr00" + x
        print(target)
    if len(x) == 8:
        target = "kplr0" + x
        print(target)

    print(x)

    # # K2 ids scraping with lightkurve package for K2scraping
    # search_result = lk.search_lightcurve(x)
    #
    # print(search_result[0])
    # exit()
    #
    # if not search_result:
    #     continue
    #
    # # for K2
    # target = search_result[0]

    # target depends on scraping type
    # general scraping of light curves from MAST archive
    # ATTENTION: change the obs_collection based on the mission
    Obs = Observations.query_criteria(target_name=target)

    if not Obs:
        continue

    print("******************")

    Prods = Observations.get_product_list(Obs[0])
    print("******************")

    yourProd = Observations.filter_products(Prods, extension='lc.fits',
                                            mrp_only=False)
    print("******************")

    print(yourProd[0])

    # changing path
    download = Observations.download_products(yourProd[0], mrp_only=False, cache=False, download_dir='A:/lightcurves')
    curr_path = download[0].table['Local Path'].value

    # change the path based on the mission
    os.rename(curr_path[0], os.path.join(pathkepler, x + '.fits'))
