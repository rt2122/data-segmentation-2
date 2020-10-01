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
    
    coords = np.array(np.where(pic != 0))
    ans = []
    while coords.shape[1] != 0:
        seed_point = tuple(coords[:, 0])
        ans.append(flood(pic, seed_point))
        pic = flood_fill(pic, seed_point, 0)
        
        coords = np.array(np.where(pic != 0))
    
    return ans

def find_centers_on_mask(mask, thr, binary=True):
    import numpy as np

    mask_binary = np.copy(mask)
    mask_binary = np.array(mask_binary >= thr, dtype=np.float32)
    
    figures = divide_figures(mask_binary)
    centers = []
    for figure in figures:
        if not binary:
            f = np.zeros_like(mask)
            f[np.where(figure)] = mask[np.where(figure)]
            centers.append(find_centroid(f))
        else:
            centers.append(find_centroid(figure))
    return centers

def clusters_in_pix(clusters, pix, nside, search_nside=None):
    import pandas as pd
    import healpy as hp
    import numpy as np
    from DS_healpix_fragmentation import radec2pix
    
    df = pd.read_csv(clusters)
    cl_pix = radec2pix(df['RA'], df['DEC'], nside)
    df = df[cl_pix == pix]
    df.index = np.arange(df.shape[0])
    if not (search_nside is None):
        search_pix = radec2pix(df['RA'], df['DEC'], search_nside)
        df['search'] = search_pix
    
    return df

def gen_pics_for_detection(ipix, model, nside=2, depth=10, step=64, size=64, 
        mask_radius=15/60, clusters_dir='/home/rt2122/Data/clusters/'):
    from DS_healpix_fragmentation import one_pixel_fragmentation, pix2radec, radec2pix
    from DS_Planck_Unet import draw_pic_with_mask, draw_pic
    import pandas as pd
    import numpy as np
    import healpy as hp
    import os
    
    true_clusters = {file[:-4] : clusters_in_pix(os.path.join(clusters_dir, file), 
                                                 ipix, 2) 
                     for file in next(os.walk(clusters_dir))[-1]}
    true_clusters['all'] = pd.concat(list(item[1] for item in true_clusters.items()))
 
    big_matr = one_pixel_fragmentation(nside, ipix, depth)
    big_pic, big_mask = draw_pic_with_mask(center=None, matr=big_matr, 
                                 mask_radius=mask_radius,
                            clusters_arr=np.array(true_clusters['all'][['RA', 'DEC']]))
    pics, matrs, masks = [], [], []
    for i in range(0, big_matr.shape[0], step):
        for j in range(0, big_matr.shape[1], step):
            pic = big_pic[i:i+size,j:j+size,:]
            mask = big_mask[i:i+size,j:j+size,:]
            matr = big_matr[i:i+size,j:j+size]
            if pic.shape[0] == size and pic.shape[1] == size:
                if np.count_nonzero(mask) > 0:
                    pics.append(pic)
                    matrs.append(matr)
                    masks.append(mask)
 
    
    ans = model.predict(np.array(pics))
    return {'true_clusters' : true_clusters,
            'pics' : pics, 'matrs' : matrs, 'masks' : masks, 'ans' : ans} 

def detect_clusters_on_pic(ans, matr, thr, binary):
    import numpy as np
    centers = find_centers_on_mask(ans, thr, binary)
    if len(centers) > 0:
        centers = np.array(centers, dtype=np.int32)
        centers = matr[centers[:,0], centers[:,1]]
    return centers

def detect_clusters(all_dict, thr, base_nside=2048, main_cat='all', max_dist=15/60, binary=True, 
        get_coords_mode=False, all_catalogs_mode=False):
    import numpy as np
    import pandas as pd
    from DS_healpix_fragmentation import pix2radec
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    masks = all_dict['masks']
    ans = all_dict['ans']
    matrs = all_dict['matrs']
    true_clusters = all_dict['true_clusters']
    sc_true_clusters = {cat : SkyCoord(ra=true_clusters[cat]['RA']*u.degree, 
                                 dec=true_clusters[cat]['DEC']*u.degree, 
                                 frame='icrs') for cat in true_clusters}
    for cat_name in true_clusters:
        true_clusters[cat_name]['found'] = False
    params = ['tp', 'fp', 'tn', 'fn']
    stat_df = dict(zip(params, [0] * len(params)))
    fp = pd.DataFrame({'RA':[], 'DEC':[]})
    fp_sc = None
    
    for i in range(len(ans)):
        centers = detect_clusters_on_pic(ans[i], matrs[i], thr, binary)
        if np.count_nonzero(masks[i]) and len(centers) == 0:
            stat_df['tn'] += 1
        if len(centers) > 0:
            centers = pix2radec(centers, nside=base_nside)
            sc = SkyCoord(ra=centers[0]*u.degree, dec=centers[1]*u.degree, frame='icrs')
            for cat in true_clusters:
                idx, d2d, _ = sc_true_clusters[cat].match_to_catalog_sky(sc)
                true_clusters[cat]['found'] = np.logical_or(d2d.degree <= max_dist,
                                                           true_clusters[cat]['found'])
                
                if fp_sc is None:
                    fp['RA'] = centers[0]
                    fp['DEC'] = centers[1]
                    fp_sc = sc
                else:
                    idx, d2d, _ = sc.match_to_catalog_sky(fp_sc)
                    fp_new = pd.DataFrame({'RA':centers[0][d2d.degree >  max_dist],
                                          'DEC':centers[1][d2d.degree >  max_dist]})
                    fp = pd.concat([fp, fp_new])
                    fp_sc = SkyCoord(ra=fp['RA']*u.degree, dec=fp['DEC']*u.degree, 
                                     frame='icrs')
    if get_coords_mode:
        return {'true_clusters' : true_clusters, 'fp' : fp}
    
    stat_df['fp'] = len(fp)

    all_stats = {}
    for cat in true_clusters:
        stat_df['tp'] = np.count_nonzero(true_clusters[cat]['found'])
        stat_df['fn'] = np.count_nonzero(np.logical_not(true_clusters[cat]['found']))
        all_stats[cat] = stat_df.copy()

    if all_catalogs_mode:
        return all_stats

    return all_stats['cat']
