def find_centroid(pic):
    from skimage.measure import moments
    import numpy as np
    
    if len(pic.shape) > 2:
        pic = np.copy(pic).reshape(list(pic.shape)[:-1])
    M = moments(pic)
    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    
    return centroid

def divide_figures(pic):
    import numpy as np
    from skimage.segmentation import flood, flood_fill
    
    coords = np.array(np.where(pic == 1))
    ans = []
    while coords.shape[1] != 0:
        seed_point = tuple(coords[:, 0])
        ans.append(flood(pic, seed_point))
        pic = flood_fill(pic, seed_point, 0)
        
        coords = np.array(np.where(pic == 1))
    
    return ans

def find_centers_on_mask(mask, thr_list=[0.8]):
    import numpy as np

    thr_dict = {}
    for thr in thr_list: 
        mask_cur = np.copy(mask) >= thr
        figures = divide_figures(mask_cur)
        centers = []
        for figure in figures:
            centers.append(find_centroid(figure))
        thr_dict[thr] = centers
        
    return thr_dict

def find_centers_on_ans(ans, matrs, thr_list=[0.8], count_blank=False):
    import numpy as np

    thr_dict = dict(zip(thr_list, [[] for i in range(len(thr_list))]))
    count_dict = dict(zip(thr_list, [0] * len(thr_list)))
    for i in range(ans.shape[0]):
        new_cen_dict = find_centers_on_mask(ans[i], thr_list=thr_list)
        for thr in thr_list:
            new_cen = new_cen_dict[thr]
            if len(new_cen) > 0:
                new_cen = np.array(new_cen).astype(np.int32)
                thr_dict[thr].extend(list(matrs[i][[new_cen[:, 0], new_cen[:, 1]]]))
            else:
                count_dict[thr] += 1
    if count_blank:
        return thr_dict, count_dict
    return thr_dict

def false_clusters(n, nside, clusters_dir, bigpixels, max_rad=1):
    import os
    import pandas as pd
    import numpy as np
    import healpy as hp
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from DS_Planck_Unet import gen_batch
    
    files = next(os.walk(clusters_dir))[-1]
    clusters = []
    for file in files:
        clusters.append(pd.read_csv(os.path.join(clusters_dir, file)))
    clusters = pd.concat(clusters)
    
    no_clusters = set(np.arange(hp.nside2npix(nside)))
    sc = SkyCoord(ra=clusters['RA']*u.degree, dec=clusters['DEC']*u.degree, frame='icrs')
    vecs = hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree, lonlat=True)
    for vec in vecs:
        no_clusters = no_clusters.difference(set(hp.query_disc(vec=vec, nside=nside, 
                            nest=True, radius=np.radians(max_rad), inclusive=False)))
        
    no_clusters = np.array(list(no_clusters))
    ang = hp.pix2ang(nside=nside, ipix=no_clusters, nest=True, lonlat=True)
    bp = hp.ang2pix(nside=2, theta=ang[0], phi=ang[1], lonlat=True, nest=True)
    no_clusters = no_clusters[np.in1d(bp, bigpixels)]
    pics, matrs = gen_batch(np.array(no_clusters), n, nside, 
                            pd.DataFrame({'RA' : [], 'DEC': []}))
    return pics, matrs


def clusters_in_pix(clusters, pix, nside):
    import pandas as pd
    import healpy as hp
    import numpy as np
    from DS_healpix_fragmentation import radec2pix
    
    df = pd.read_csv(clusters)
    cl_pix = radec2pix(df['RA'], df['DEC'], nside)
    df = df[cl_pix == pix]
    df.index = np.arange(df.shape[0])
    
    return df


def scan_pix(clusters, model, ipix, nside=2, depth=10, thr=0.8, min_dist=5/60, 
             step=64, size=64, n_false=None, search_nside=256):
    from DS_healpix_fragmentation import one_pixel_fragmentation, pix2radec, radec2pix
    from DS_Planck_Unet import draw_pic_with_mask, draw_pic
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import pandas as pd
    import numpy as np
    import healpy as hp
    from tensorflow.keras import backend as K
    from tqdm.notebook import tqdm
    
    big_matr = one_pixel_fragmentation(nside, ipix, depth)
    
    true_clusters = clusters_in_pix(clusters, ipix, nside)
    if n_false is None:
        n_false = len(true_clusters)
    
    pics, matrs = [], []
    
    for i in range(0, big_matr.shape[0] - step, step):
        for j in range(0, big_matr.shape[1], step):
            matr = big_matr[i:i+size,j:j+size]
            if matr.shape[0] == size and matr.shape[1] == size:
                pic = draw_pic(matr)
                pics.append(pic)
                matrs.append(matr)
    pics = np.array(pics)
    ans = model.predict(pics)
    ans = np.array(ans)
    found_clusters = find_centers_on_ans(ans, matrs, thr)
    all_found = len(found_clusters)
    theta, phi = hp.pix2ang(ipix=found_clusters, nest=True, nside=nside*2**depth,
                           lonlat=True)    
    sc_true = SkyCoord(ra=true_clusters['RA']*u.degree, 
                       dec=true_clusters['DEC']*u.degree, frame='icrs')
    sc_found = SkyCoord(l=theta*u.degree,
                       b=phi*u.degree, frame='galactic')
    
    idx, d2d, _ = sc_found.match_to_catalog_sky(sc_true)
    tp = np.count_nonzero(d2d.degree <= min_dist)
    fn = true_clusters.shape[0] - tp
    #----test false clusters----#
    pics, matrs = false_clusters(n_false, search_nside, '/home/rt2122/Data/clusters/',
                                [ipix])
    pics = np.array(pics)
    ans = model.predict(pics)
    ans = np.array(ans)
    found_clusters, tn = find_centers_on_ans(ans, matrs, thr, count_blank=True)
    all_found += len(found_clusters)
    fp = len(found_clusters)
    
    
    
    res_table = pd.DataFrame({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 
                              'all_found' : all_found,
                             'min_dist' : d2d.degree.min(), 'pix2' : ipix}, index=[0])
    return res_table
