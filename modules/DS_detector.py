def find_centroid(pic):
    from skimage.measure import moments
    import numpy as np
    
    if len(pic.shape) > 2:
        pic = np.copy(pic).reshape(list(pic.shape)[:-1])
    M = moments(pic)
    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    
    return centroid

def pack_all_catalogs(cat_dir='/home/rt2122/Data/clusters/'):
    import os
    import numpy as np
    import pandas as pd
    
    all_cats = []
    files = next(os.walk(cat_dir))[-1]
    for file in files:
        df = pd.read_csv(os.path.join(cat_dir, file))
        df['catalog'] = file[:-4]
        all_cats.append(df)
    all_cats = pd.concat(all_cats, ignore_index=True)
    
    return all_cats

def get_radius(figure, center):
    import numpy as np
    from skimage.filters import roberts
    center = np.array(center)
    
    edge = np.where(roberts(figure) != 0)
    min_rad = figure.shape[0] * 2
    max_rad = -1
    
    for point in zip(*edge):
        rad = np.linalg.norm(center - np.array(point))
        min_rad = min(min_rad, rad)
        max_rad = max(max_rad, rad)
    
    return min_rad, max_rad

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
    areas = []
    min_rad = []
    max_rad = []
    min_pred = []
    max_pred = []
    for figure in figures:
        f = np.zeros_like(mask)
        f[np.where(figure)] = mask[np.where(figure)]

        if not binary:
            centers.append(find_centroid(f))
        else:
            centers.append(find_centroid(figure))
        
        areas.append(np.count_nonzero(figure))
        rads = get_radius(figure[:,:,0], centers[-1])
        min_rad.append(rads[0])
        max_rad.append(rads[1])
        min_pred.append(np.partition(list(set(f.flatten())), 1)[1])
        max_pred.append(f.max())

    return {'centers' : np.array(centers), 'areas' : np.array(areas), 
            'min_rad' : np.array(min_rad), 'max_rad' : np.array(max_rad),
           'min_pred': np.array(min_pred), 'max_pred' : np.array(max_pred)}

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


def gen_pics_for_detection(ipix, model, big_nside=2, step=64, size=64, depth=10, 
        mask_radius=15/60, clusters_dir='/home/rt2122/Data/clusters/'):
    from DS_healpix_fragmentation import one_pixel_fragmentation, pix2radec, radec2pix
    from DS_Planck_Unet import draw_pic_with_mask, draw_pic
    import pandas as pd
    import numpy as np
    import healpy as hp
    import os
    
    true_clusters = pack_all_catalogs(clusters_dir)
    clusters_pix = radec2pix(true_clusters['RA'], true_clusters['DEC'], 2)
    true_clusters = true_clusters[clusters_pix == ipix]
    true_clusters.index = np.arange(true_clusters.shape[0])
 
    big_matr = one_pixel_fragmentation(big_nside, ipix, depth)
    big_pic, big_mask = draw_pic_with_mask(center=None, matr=big_matr, 
                            mask_radius=mask_radius,
                            clusters_arr=np.array(true_clusters[['RA', 'DEC']]))
    
    pics, matrs, masks = [], [], []
    for i in range(0, big_matr.shape[0], step):
        for j in range(0, big_matr.shape[1], step):
            pic = big_pic[i:i+size,j:j+size,:]
            mask = big_mask[i:i+size,j:j+size,:]
            matr = big_matr[i:i+size,j:j+size]
            
            if pic.shape == (size, size, pic.shape[-1]):
                if np.count_nonzero(mask) > 0:
                    pics.append(pic)
                    matrs.append(matr)
                    masks.append(mask)
 
    
    ans = model.predict(np.array(pics))
    return {'true_clusters' : true_clusters,
            'pics' : pics, 'matrs' : matrs, 'masks' : masks, 'ans' : ans} 

def detect_clusters_on_pic(ans, matr, thr, binary):
    import numpy as np
    dd = find_centers_on_mask(ans, thr, binary)
    if len(dd['centers']) > 0:
        centers = np.array(dd['centers'], dtype=np.int32)
        dd['centers'] = matr[centers[:,0], centers[:,1]]
    return dd

def detect_clusters(all_dict, thr, base_nside=2048, tp_dist=5/60, 
                        fp_dist=15/60, binary=False, ret_coords=True,
                        match_before_merge=True):
    import numpy as np
    import pandas as pd
    from DS_healpix_fragmentation import pix2radec
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    masks = all_dict['masks']
    ans = all_dict['ans']
    matrs = all_dict['matrs']
    true_clusters = all_dict['true_clusters']
    true_clusters['found'] = False
    true_clusters_sc = SkyCoord(ra=true_clusters['RA']*u.degree, 
                                dec=true_clusters['DEC']*u.degree, frame='icrs')
    
    res_cat = pd.DataFrame({'RA' : [], 'DEC' : [], 'area' : [], 
                      'min_rad' : [], 'max_rad' : [],
                      'min_pred' : [], 'max_pred' : [], 
                      'tRA':[], 'tDEC' : []})
    res_cat['status'] = ''
    res_cat['catalog'] = ''
    res_cat_sc = None
    
    params = ['tp', 'fp', 'tn', 'fn']
    stat_df = dict(zip(params, [0] * len(params)))
    
    for i in range(len(ans)):
        dd_pic = detect_clusters_on_pic(ans[i], matrs[i], thr, binary)
        centers = dd_pic['centers']
        
        if np.count_nonzero(masks[i]) and len(centers) == 0:
            stat_df['tn'] += 1
        
        if len(centers) > 0: 
            centers = pix2radec(centers, nside=base_nside)
            sc = SkyCoord(ra=centers[0]*u.degree, 
                          dec=centers[1]*u.degree, frame='icrs')

            res_cat_new = pd.DataFrame({'RA':centers[0],
                                  'DEC':centers[1],
                                  'area' : dd_pic['areas'],          
                'min_rad' : dd_pic['min_rad'],
                'max_rad' : dd_pic['max_rad'],
                'min_pred' : dd_pic['min_pred'],
                'max_pred' : dd_pic['max_pred'],
                                      })
            res_cat_new['tRA'] = np.NaN
            res_cat_new['tDEC'] = np.NaN
            res_cat_new['status'] = 'fp'
            res_cat_new['catalog'] = ''
            if match_before_merge:
                idx, d2d, _  = sc.match_to_catalog_sky(true_clusters_sc)
                tp_match = d2d.degree <= tp_dist
                res_cat_new['status'].iloc[tp_match] = 'tp'
                res_cat_new['catalog'].iloc[tp_match] = np.array(
                        true_clusters['catalog'][idx[tp_match]])
                res_cat_new['tRA'].iloc[tp_match] = np.array(true_clusters['RA'][idx[tp_match]])
                res_cat_new['tDEC'].iloc[tp_match] = np.array(true_clusters['DEC'][idx[tp_match]])
                true_clusters['found'].iloc[idx[tp_match]] = True
                
            if res_cat_sc is None:
                res_cat = res_cat_new
                res_cat_sc = sc
            else: 
                '''
                res_cat_new = pd.DataFrame({'RA':centers[0][res_cat_new_idx],
                                  'DEC':centers[1][res_cat_new_idx],
                                  'area' : dd_pic['areas'][res_cat_new_idx],          
                'min_rad' : dd_pic['min_rad'][res_cat_new_idx],
                'max_rad' : dd_pic['max_rad'][res_cat_new_idx],
                'min_pred' : dd_pic['min_pred'][res_cat_new_idx],
                'max_pred' : dd_pic['max_pred'][res_cat_new_idx],
                                      })'''
                
                res_cat_new_fp = res_cat_new[res_cat_new['status'] == 'fp']
                res_cat_new_fp.index = np.arange(len(res_cat_new_fp))
                res_cat_new_tp = res_cat_new[res_cat_new['status'] == 'tp']
                res_cat_new_tp.index = np.arange(len(res_cat_new_tp))

                sc_fp = SkyCoord(ra=np.array(res_cat_new_fp['RA'])*u.degree, 
                          dec=np.array(res_cat_new_fp['DEC'])*u.degree, frame='icrs')

                idx, d2d, _ = sc_fp.match_to_catalog_sky(res_cat_sc)
                res_cat_new_fp = res_cat_new_fp[d2d.degree > fp_dist]
                res_cat_new_fp.index = np.arange(len(res_cat_new_fp))

                #res_cat_new_drop = d2d.degree <=  fp_dist
                res_cat = pd.concat([res_cat, res_cat_new_tp, res_cat_new_fp], ignore_index=True)
                res_cat_sc = SkyCoord(ra=res_cat['RA']*u.degree, dec=res_cat['DEC']*u.degree, 
                                 frame='icrs')
        
    
    if not match_before_merge:
        idx, d2d, _ = res_cat_sc.match_to_catalog_sky(true_clusters_sc)
        matched_idx = d2d.degree <= tp_dist
        res_cat['status'].iloc[matched_idx] = 'tp'
        res_cat['catalog'].iloc[matched_idx] = np.array(
            true_clusters['catalog'][idx[matched_idx]])
        res_cat['status'].iloc[np.logical_not(matched_idx)] = 'fp'
        
        true_clusters['found'].iloc[idx[matched_idx]] = True
    else:
        res_cat_tp = res_cat[res_cat['status'] == 'tp']
        res_cat_tp = res_cat_tp.drop_duplicates(subset=['tRA', 'tDEC'])
        res_cat = pd.concat([res_cat[res_cat['status'] != 'tp'], res_cat_tp], ignore_index=True)

    not_found = true_clusters[np.logical_not(true_clusters['found'])]
    
    fn = pd.DataFrame({'RA' : not_found['RA'], 'DEC' : not_found['DEC'], 
                      'catalog' : not_found['catalog'], 
                       'status' : ['fn'] * len(not_found)})
    
    res_cat = pd.concat([res_cat, fn], ignore_index=True)
    if ret_coords:
        return res_cat
    all_stats = {}
    stat_df['fp'] = np.count_nonzero(res_cat['status'] == 'fp')
    for cat in set(res_cat['catalog']):
        if len(cat) > 0:
            cur_cat = res_cat[res_cat['catalog'] == cat]
            cur_cat.index = np.arange(len(cur_cat))
            stat_df['tp'] = np.count_nonzero(cur_cat['status'] == 'tp')
            stat_df['fn'] = np.count_nonzero(cur_cat['status'] == 'fn')
            all_stats[cat] = pd.DataFrame(stat_df, index=[0])
    for cat in all_stats:
        all_stats[cat]['catalog'] = cat
    all_stats = pd.concat([all_stats[cat] for cat in all_stats], ignore_index=True)
    return all_stats 

