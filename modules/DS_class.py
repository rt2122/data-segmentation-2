def_class_len = 50

def radec2pixvec_dist(cen, pixv, nside):
    import healpy as hp
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    sc_cen = SkyCoord(ra=cen[0]*u.degree, dec=cen[1]*u.degree, frame='icrs')
    l, b = hp.pix2ang(nside=nside, ipix=pixv, nest=True, lonlat=True)
    sc_vec = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
    return sc_cen.separation(sc_vec).degree


def get_planck_by_pix(ipix, planck_nside=2048, 
                      dirname='/home/rt2122/Data/Planck/normalized/'):
    import numpy as np
    import os
    
    files = sorted(next(os.walk(dirname))[-1])
    res = []
    for i_f, file in enumerate(files):
        i_s = np.load(os.path.join(dirname, file))
        res.append(i_s[ipix])
    return np.stack(res, axis=1)

def radec2line_class(coords, radius=7.5/60, class_len=50, planck_nside=2048):
    import numpy as np
    import healpy as hp
    from DS_healpix_fragmentation import radec2vec 
    
    vec = radec2vec(*coords)
    closest_pix = hp.query_disc(nside=planck_nside, vec=vec, radius=np.radians(radius))
    dists = radec2pixvec_dist(coords, closest_pix, nside=planck_nside)
    #sort pix by dist
    arr = list(np.stack([closest_pix, dists]).T)
    arr = sorted(arr, key=lambda x:x[1])
    
    closest_pix = np.array(arr, dtype=np.int64)[:class_len,0]
    res = get_planck_by_pix(closest_pix)
    return res 

def gen_data_for_class(tp_coef, n, cats, pix2, dirname='/home/rt2122/Data/clusters/',
                      fp_file='/home/rt2122/Data/class/fp_coords/fp_pz_pnz_act.csv',
                      radius=7.5/60, class_len=50):
    import os
    import numpy as np
    import pandas as pd
    from DS_healpix_fragmentation import radec2pix
    
    df_cat = pd.concat([pd.read_csv(os.path.join(dirname, cat + '.csv')) 
                        for cat in cats], ignore_index=True)
    df_cat['pix2'] = radec2pix(df_cat['RA'], df_cat['DEC'], 2)
    df_cat = df_cat[np.in1d(df_cat['pix2'], pix2)]
    df_cat.index = np.arange(len(df_cat))
    
    n_tp = int(tp_coef * n)
    n_fp = n - n_tp
    
    tp_df = df_cat.sample(n=n_tp)[['RA', 'DEC']]
    fp_df = pd.read_csv(fp_file)
    fp_df['pix2'] = radec2pix(fp_df['RA'], fp_df['DEC'], 2)
    fp_df = fp_df[np.in1d(fp_df['pix2'], pix2)].sample(n=n_fp)[['RA', 'DEC']]
    tp_df['y'] = True
    fp_df['y'] = False
    df = pd.concat([tp_df, fp_df], ignore_index=True)
    
    X = np.stack([radec2line_class([df['RA'].iloc[i], df['DEC'].iloc[i]], 
                                radius=radius, class_len=class_len).flatten()
               for i in range(len(df))])
    
    
    y = np.array(df['y'])
    return X, y

def detected_cat2class(cat_df, clf, label='class', radius=7.5/60, class_len=50, nn_mode=False):
    import numpy as np
    import pandas as pd
    from DS_healpix_fragmentation import radec2pix
    
    X = np.stack([radec2line_class([cat_df['RA'].iloc[i], cat_df['DEC'].iloc[i]], 
                                radius=radius, class_len=class_len).flatten()
               for i in range(len(cat_df))])
    if nn_mode:
        ans = clf.predict(X)
        cat_df[label] = ans
    else:
        ans = clf.predict_proba(X)
        cat_df[label] = ans[:, 1]
    return cat_df
