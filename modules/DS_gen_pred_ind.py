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

def gen_scan_pix(ipix, model, big_nside=2, depth=10, step=8, size=64, 
                 planck_dir='/home/rt2122/Data/Planck/normalized'):
    from DS_healpix_fragmentation import one_pixel_fragmentation
    from DS_Planck_Unet import draw_pic
    
    big_matr = one_pixel_fragmentation(big_nside, ipix, depth)
    big_pic = draw_pic(big_matr, dirname=planck_dir)
    
    starts = []
    for k in range(2):
        x_st = [i for i in range(0, big_matr.shape[k], step) 
                if i + size <= big_matr.shape[k]] + [big_matr.shape[k] - size]
        starts.append(x_st) 

    pics = []
    coords = []
    for x in starts[0]:
        for y in starts[1]:
            pics.append(big_pic[x:x+size, y:y+size, :])
            coords.append([x, y])
    masks = model.predict(np.array(pics))
    
    big_mask = np.zeros(big_matr.shape)
    coef = np.zeros(big_matr.shape)
    for i, [x, y] in enumerate(coords):
        big_mask[x:x+64, y:y+64] = masks[i][:,:,0]
        coef[x:x+64, y:y+64] += 1
    
    big_mask /= coef
    return big_mask, big_matr

def full_scan(model_name, scan_name, step=8, planck_dir='/home/rt2122/Data/Planck/normalized'):
    from tqdm.notebook import tqdm
    from DS_Planck_Unet import load_planck_model
    import os
    import healpy as hp
    import numpy as np
    
    model = load_planck_model(model_name)
    scan = np.zeros(hp.nside2npix(2**11))
    for i in tqdm(range(48)):
        mask, matr = gen_scan_pix(i, model, step=step, planck_dir=planck_dir)
        mask = mask.flatten()
        matr = matr.flatten()
        scan[matr] = mask
    np.save(scan_name, scan)
