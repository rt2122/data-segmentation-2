def find_centroid(pic):
    from skimage.measure import moments
    import numpy as np
    #print(pic.shape)
    
    if len(pic.shape) > 2:
        pic = np.copy(pic).reshape(list(pic.shape)[:-1])
    M = moments(pic)
    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    
    return centroid

def divide_figures(pic):
    import numpy as np
    from skimage.segmentation import flood, flood_fill
    #print(pic.shape)
    
    coords = np.array(np.where(pic != 0))
    ans = []
    while coords.shape[1] != 0:
        seed_point = tuple(coords[:, 0])
        ans.append(flood(pic, seed_point))
        pic = flood_fill(pic, seed_point, 0)
        
        coords = np.array(np.where(pic != 0))
    
    return ans

def find_centers_on_mask(mask, thr_list=[0.8]):
    import numpy as np

    thr_dict = {}
    for thr in thr_list: 
        mask_cur = np.copy(mask)
        #mask_cur = mask_cur[mask_cur >= thr]
        mask_cur = np.array(mask_cur >= thr, dtype=np.float32)
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

def proc_found_clusters(pics, matrs, model, nside=2, depth=10, thr_list=[0.8], 
                        true_clusters=None, 
                        true_mode=True, min_dist=5/60):
    import numpy as np
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    
    pics = np.array(pics)
    ans = model.predict(pics)
    ans = np.array(ans)
    found_clusters_dict, count_blank = \
        find_centers_on_ans(ans, matrs, thr_list, count_blank=True)
    
    sc_true = None
    if not(true_clusters is None):
        sc_true = SkyCoord(ra=true_clusters['RA']*u.degree, 
                       dec=true_clusters['DEC']*u.degree, frame='icrs')
    
    df = pd.DataFrame({'p':[0 for i in range(len(thr_list))], 
                       'n':[0 for i in range(len(thr_list))], 
                       'thr': [0 for i in range(len(thr_list))], 
                       'min_dist': [0 for i in range(len(thr_list))],
                       'all_found': [0 for i in range(len(thr_list))]})
    max_tp = []
    max_fp = []
    for idx in range(len(thr_list)):
        thr = thr_list[idx]
        all_found = len(found_clusters_dict[thr])
        if all_found > 0:
            theta, phi = hp.pix2ang(ipix=np.array(found_clusters_dict[thr]), nest=True, 
                            nside=nside*2**depth, lonlat=True)

            sc_found = SkyCoord(l=theta*u.degree,
                               b=phi*u.degree, frame='galactic')
        if true_mode: 
            p = 0
            fp = None
            if all_found > 0:
                cluster_idx, d2d, _ = sc_found.match_to_catalog_sky(sc_true)
                fp = d2d.degree > min_dist
                tp_idx = set(cluster_idx[d2d.degree <= min_dist])
                p = len(tp_idx)
                df['min_dist'].iloc[idx] = d2d.degree.min()

            n = true_clusters.shape[0] - p

            if p >= len(max_tp):
                max_tp = true_clusters.iloc[list(tp_idx)]
            if all_found - p >= len(max_fp):
                max_fp = np.array(found_clusters_dict[thr])[fp]


        else:
            p = all_found
            n = count_blank[thr]

            if p > len(max_fp):
                max_fp = np.array(found_clusters_dict[thr])
            
        df['p'].iloc[idx] = p
        df['n'].iloc[idx] = n
        df['thr'].iloc[idx] = thr
        df['all_found'].iloc[idx] = all_found
        
    return df, max_tp, max_fp


def scan_pix(clusters, model, ipix, nside=2, depth=10, thr_list=[0.8], min_dist=5/60, 
             step=64, size=64, n_false=None, search_nside=256, big_mask_radius=15/60):
    from DS_healpix_fragmentation import one_pixel_fragmentation, pix2radec, radec2pix
    from DS_Planck_Unet import draw_pic_with_mask, draw_pic
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
    blank_pics, blank_matrs = [], []
    
    for i in range(0, big_matr.shape[0], step):
        for j in range(0, big_matr.shape[1], step):
            matr = big_matr[i:i+size,j:j+size]
            if matr.shape[0] == size and matr.shape[1] == size:
                pic, mask = draw_pic_with_mask(matr=matr, 
                            clusters_arr=np.array(true_clusters[['RA', 'DEC']]),
                                center=None, mask_radius=big_mask_radius)
                if np.count_nonzero(mask) > 0:
                    pics.append(pic)
                    matrs.append(matr)
                else:
                    blank_pics.append(pic)
                    blank_matrs.append(matr)
    
    #----test true clusters----#
    res_t, max_tp, max_fp = proc_found_clusters(pics, matrs, model, nside=nside, depth=depth,
                                true_clusters=true_clusters, 
                                true_mode=True, min_dist=min_dist, thr_list=thr_list)
    #----test false clusters----#
    res_f, _, max_fp_a = proc_found_clusters(blank_pics, blank_matrs, model, nside=nside,
                                depth=depth, true_mode=False, min_dist=min_dist,
                               thr_list=thr_list)
    max_fp = np.hstack([max_fp, max_fp_a])
    res_table = pd.DataFrame({'tp' : res_t['p'], 'tn' : res_f['n'], 
                             'fp' : res_t['all_found'] - res_t['p'] + res_f['p'],
                             'fn' : res_t['n'], 'thr' : thr_list, 
                              'pix2': [ipix for i in range(len(thr_list))]})
    
    return res_table, max_tp, max_fp

def scan_none(pix2, model, clusters_dir='/home/rt2122/Data/clusters/', 
              thr_list=[0.1], size=64, depth=10, 
              planck_nside=2048, clusters_dist=15/60, ret_coords=False):
    from DS_healpix_fragmentation import one_pixel_fragmentation
    from DS_Planck_Unet import draw_pic
    import os
    import healpy as hp
    import pandas as pd
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    matr = one_pixel_fragmentation(2, pix2, depth)
    true_clusters = {file[:-4] : clusters_in_pix(os.path.join(clusters_dir, file), pix2, 2) 
                     for file in next(os.walk(clusters_dir))[-1]}
    true_clusters['all'] = pd.concat(list(item[1] for item in true_clusters.items()))
    
    pic = draw_pic(matr)
    ans = model.predict(np.array([pic]))
    
    thr_dict = find_centers_on_ans(ans, [matr], thr_list)
    sc_true = {name : SkyCoord(ra=true_clusters[name]['RA']*u.degree, 
                               dec=true_clusters[name]['DEC']*u.degree,
                       frame='icrs') for name in true_clusters}
    coords = {}
    df = pd.DataFrame({
        'tp_pz':[0] * len(thr_list), 
        'tp_pnz':[0] * len(thr_list), 
        'tp_m':[0] * len(thr_list), 
        'tn':[0] * len(thr_list), 
        'fp':[0] * len(thr_list), 
        'fn_pz':[0] * len(thr_list), 
        'fn_pnz':[0] * len(thr_list), 
        'fn_m':[0] * len(thr_list), 
                   'thr': thr_list, 
                   'min_dist':[0] * len(thr_list),
                   'all_found':[0] * len(thr_list)})
    for i in range(len(thr_list)):
        thr = thr_list[i]
        df['all_found'][i] = len(thr_dict[thr])
        
        if len(thr_dict[thr]) > 0:
            theta, phi = hp.pix2ang(nside=planck_nside, nest=True, ipix=thr_dict[thr], 
                                lonlat=True)
            sc_found = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
        
            for name, sm_name in [('planck_z', 'pz'), 
                                 ('planck_no_z', 'pnz'), ('mcxcwp', 'm')]:

                idx, d2d, _ = sc_found.match_to_catalog_sky(sc_true[name])
                idx = np.unique(idx[d2d.degree <= clusters_dist])
                tp = np.count_nonzero(d2d.degree <= clusters_dist)
                df['tp_'+sm_name][i] = tp 
                df['fn_'+sm_name][i] = len(true_clusters[name]) - tp
                coords[name] = true_clusters[name].iloc[idx]
                
            idx, d2d, _ = sc_found.match_to_catalog_sky(sc_true['all'])
            df['fp'][i] = np.count_nonzero(d2d.degree > clusters_dist)
            found = pd.DataFrame({
                'RA':sc_found.icrs.ra.degree,
                'DEC':sc_found.icrs.dec.degree})
            coords['fp'] = found[d2d.degree > clusters_dist]
        
        else:
            df['fn_pz'][i] = len(true_clusters['planck_z'])
            df['fn_pnz'][i] = len(true_clusters['planck_no_z'])
            df['fn_m'][i] = len(true_clusters['mcxcwp'])
    
    if ret_coords:
        return coords
    return df
        
