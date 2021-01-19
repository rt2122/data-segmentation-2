def radec2pix(ra, dec, nside):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    import numpy as np

    sc = SkyCoord(ra=np.array(ra)*u.degree, dec=np.array(dec)*u.degree, frame='icrs')
    return hp.ang2pix(nside, sc.galactic.l.degree, sc.galactic.b.degree, 
                                  nest=True, lonlat=True)

def get_df(filename, mode='fits'):
    import pandas as pd
    import gzip
    import pickle
    from astropy.io import fits
    from astropy.table import Table
    
    if mode == 'fits':
        with fits.open(filename) as hdul:
            tbl = Table(hdul[1].data)
            tbl = tbl['srcID', 'RA', 'DEC']
            df = tbl.to_pandas()
            return df
    
    if mode == 'gz_pkl':
         with gzip.open(filename, 'rb') as f:
            df = pickle.load(f)[['srcID_', 'RA_', 'DEC_']]
            df.rename(columns={'srcID_' : 'srcID', 'RA_' : 'RA', 'DEC_' : 'DEC'}, inplace=True)
            return df

def radec2pred_ind(scan_name, input_name, output_name, mode='fits'):
    import os
    import numpy as np
    import pandas as pd
    
    in_df = get_df(input_name, mode=mode)
    in_df['healpix'] = radec2pix(in_df['RA'], in_df['DEC'], 2**11)
    
    scan = np.load(scan_name)
    in_df['pred_ind'] = scan[in_df['healpix']]
    
    in_df.index.name='index'
    in_df.to_csv(output_name)
