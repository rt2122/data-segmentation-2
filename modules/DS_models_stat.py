def stat_split_cats(files, big_pix=list(range(48))):
    import pandas as pd
    import numpy as np
    from DS_healpix_fragmentation import radec2pix

    res_df = []
    comp_df = []
    for model in files:
        df = pd.read_csv(files[model])
        df = df.iloc[np.in1d(radec2pix(df['RA'], df['DEC'], 2), big_pix)]
        df.index = np.arange(len(df))
        line = {}
        line_c = {}
        for cat in ['planck_z', 'planck_no_z', 'mcxcwp', 'actwp']:
            cur_df = df[df['catalog'] == cat]
            cur_df.index = np.arange(len(cur_df))
            line_c[cat] = np.count_nonzero(cur_df['status'] == 'tp')
            line[cat] = line_c[cat] / len(cur_df)
        line['fp'] = np.count_nonzero(df['status'] == 'fp')
        res_df.append(pd.DataFrame(line, index=[model]))
        comp_df.append(pd.DataFrame(line_c, index=[model]))
    res_df = pd.concat(res_df)
    comp_df = pd.concat(comp_df)
    return res_df, comp_df

def cut_cat(df, dict_cut = {}, 
           big_pix=None):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    from DS_healpix_fragmentation import radec2pix
 
    sc = SkyCoord(ra=np.array(df['RA'])*u.degree, 
                  dec=np.array(df['DEC'])*u.degree, frame='icrs')
    df['b'] = sc.galactic.b.degree
    for prm in dict_cut:
        if prm == 'b':
            df = df[np.abs(df[prm]) >= dict_cut[prm][0]]
            df = df[np.abs(df[prm]) < dict_cut[prm][1]]
            df.index = np.arange(len(df))
        else: 
            df = df[df[prm] >= dict_cut[prm][0]]
            df = df[df[prm] < dict_cut[prm][1]]
            df.index = np.arange(len(df))
    
    if not (big_pix is None):
        pix2 = radec2pix(df['RA'], df['DEC'], 2)
        df = df[np.in1d(pix2, big_pix)]
        df.index = np.arange(len(df))
    
    return df

def stat_orig_cats(det_cats_dict, true_cats_dir, 
        dict_cut = {}, big_pix = None, match_dist=5/60, n_try=200):
    import os
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from DS_data_transformation import calc_error
    import numpy as np
    import pandas as pd
    
    true_cats_files = next(os.walk(true_cats_dir))[-1]
    true_cats_files = [os.path.join(true_cats_dir, file) for file in true_cats_files]
    
    true_cats = {os.path.splitext(os.path.basename(file))[0] : pd.read_csv(file) for file in true_cats_files}
    det_cats = {name : 
                pd.read_csv(det_cats_dict[name]) for name in det_cats_dict}
    
    comp_df = []
    recall_df = []
    
    for name in det_cats:
        df = det_cats[name]
        df = df[df['status'] != 'fn']
        df.index = np.arange(len(df))
        if 'b' in dict_cut:
            det_cats[name] = cut_cat(df, {'b' : dict_cut['b']}, big_pix)
        else:
            det_cats[name] = cut_cat(df, {}, big_pix)
    for name in true_cats:
        true_cats[name] = cut_cat(true_cats[name], dict_cut, big_pix)
    
    for det_name in det_cats:
        det = det_cats[det_name]
        line = {}
        line_r = {}

        det_sc = SkyCoord(ra=np.array(det['RA'])*u.degree, 
                      dec=np.array(det['DEC'])*u.degree, frame='icrs') 

        for tr_name in true_cats: 
            tr = true_cats[tr_name]
            tr_sc = SkyCoord(ra=np.array(tr['RA'])*u.degree, 
                          dec=np.array(tr['DEC'])*u.degree, frame='icrs')
            
            idx, d2d, _ = tr_sc.match_to_catalog_sky(det_sc)
            matched = d2d.degree <= match_dist
            
            line[tr_name] = np.count_nonzero(matched)
            line[tr_name+'_err'], line[tr_name+'_std'] = calc_error(det, tr, n_try=n_try)

            line_r[tr_name] = line[tr_name] / len(tr)
            
        line['all'] = len(det)
        line['fp'] = np.count_nonzero(det['status'] == 'fp')
        line_r['fp'] = line['fp']
        line_r['all'] = line['all']
        comp_df.append(pd.DataFrame(line, index=[det_name]))
        recall_df.append(pd.DataFrame(line_r, index=[det_name]))
        line = {}
    
    for tr_name in true_cats: 
        line[tr_name] = len(true_cats[tr_name])
        line[tr_name+'_err'] = 0
    line['fp'] = 0
    line['all'] = 0
    comp_df.append(pd.DataFrame(line, index=['all']))
    
    comp_df = pd.concat(comp_df)
    recall_df = pd.concat(recall_df)
    return comp_df, recall_df
