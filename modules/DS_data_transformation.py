def fits2df(fitsname, filtered=False, one_col=None):
    from astropy.io import fits
    from astropy.table import Table
    import numpy as np
    
    df = None
    with fits.open(fitsname) as hdul:
        if not (one_col is None):
            return np.array(hdul[1].data[one_col])
        tbl = Table(hdul[1].data)
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        df = tbl[names].to_pandas()
    if filtered:
        df['phot_is_star_gaia'] = 0 
        sn2 = 4**2
        f = ~((df['PARALLAX'].isna()) | (df['PARALLAX']==0))
        df.loc[f,'phot_is_star_gaia'] = df.loc[f,'phot_is_star_gaia'] + \
            ((df.loc[f,'PARALLAX']**2) * df.loc[f,'PARALLAX_IVAR'] > sn2).astype(int)
        f = ~((df['PMRA'].isna()) | (df['PMRA']==0))
        df.loc[f,'phot_is_star_gaia'] = df.loc[f,'phot_is_star_gaia'] + \
            ((df.loc[f,'PMRA']**2) * df.loc[f,'PMRA_IVAR'] > sn2).astype(int)
        f = ~((df['PMDEC'].isna()) | (df['PMDEC']==0))
        df.loc[f,'phot_is_star_gaia'] = df.loc[f,'phot_is_star_gaia'] + \
            ((df.loc[f,'PMDEC']**2) * df.loc[f,'PMDEC_IVAR'] > sn2).astype(int)
        df = df[df['phot_is_star_gaia']==0]
        df.index = np.arange(df.shape[0])
    return df

def pic2fits(pic, wcs, fitsname):
    from astropy.io import fits
    import numpy as np

    hdul = None
    if wcs is None:
        hdul = fits.HDUList([fits.PrimaryHDU(), 
            fits.ImageHDU(np.stack([pic[:,:,i] for i in range(pic.shape[-1])]))]) 
    else:       
        hdul = fits.HDUList([fits.PrimaryHDU(), 
            fits.ImageHDU(np.stack([pic[:,:,i] for i in range(pic.shape[-1])]), 
                         header=wcs.to_header())])

    hdul.writeto(fitsname)

def show_pic(pic, projection=None, label = 'label', figsize=(10, 10), vmin=0, vmax=1, 
        slices=None):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1,0.1,0.8,0.8], projection=projection)
    plt.xlabel(label)
    if not (projection is None):
        ra = ax.coords[0]
        ra.set_format_unit('degree')

    im = ax.imshow(pic, cmap=plt.get_cmap('viridis'), 
                   interpolation='nearest', vmin=0, vmax=1)


def nth_max(array, n):
    import numpy as np
    return np.partition(array, -n)[-n]

def n_max_flux(flux, n):
    import numpy as np
    max_n = nth_max(np.array(flux), n)
    return flux >= max_n

def n_max_flux_df(df, n, ch):
    import numpy as np
    if type(ch) == type(''):
        ch = df[ch]
    else:
        ch = df[ch].sum(axis=1)
    df = df[n_max_flux(ch, n)]
    df.index = np.arange(df.shape[0])
    return df

def normalize(pic):
    import numpy as np

    pic = np.copy(pic)
    for i in range(pic.shape[-1]):
        pic[:,:,i] -= np.mean(pic[:,:,i])
        pic[:,:,i] /= np.std(pic[:,:,i])
    return pic


def draw_df(data, base, figsize=(8, 6), grids=True, xgrid=None, ygrid=None,
               save=False, invert=True, comment='', comm_coords=(0.6, 0.1)):
    from matplotlib import pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(1, figsize=figsize)
    colors = 'bgrcmykw'
    if invert:
        plt.gca().invert_xaxis()
    max_y = 0
    min_y = 100
    for c, label in zip(colors[:len(data)], data):
        line, = ax.plot(base, data[label], c+'o-')
        line.set_label(label)
        max_y = max(max_y, max(data[label]))
        min_y = min(min_y, min(data[label]))
    ax.legend()
    if grids:
        if xgrid is None:
            ax.set_xticks(base)
        else:
            ax.set_xticks(xgrid)
            
        if ygrid is None:
            ax.set_yticks(np.arange(min_y, max_y, (max_y-min_y) / 10))
        else:
            ax.set_yticks(ygrid)
        #ax.grid(True)
        plt.grid(b=True, which='major', color='#666666', linestyle=':')
    plt.text(*comm_coords, comment)
    if save:
        plt.savefig('pic.png')
    plt.show()


def get_prm(prm, s, w='\d'):
    import re
    return re.search(prm + r'(' + w + '+)', s)[0][len(prm):]

def calc_error(det_cat, true_cat, shift=15/60, match_dist=5/60, n_try=20, seed=0):
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    error = []
    np.random.seed(seed)
    for _ in range(n_try):
        det_sc = SkyCoord(ra=np.array(det_cat['RA']) * u.degree, 
                          dec=np.array(det_cat['DEC']) * u.degree, frame='icrs')
        angles = np.random.randint(0, 360, len(det_cat))
        det_sc = det_sc.directional_offset_by(angles*u.degree, shift)

        true_sc = SkyCoord(ra=np.array(true_cat['RA']) * u.degree, 
                           dec=np.array(true_cat['DEC']) * u.degree, frame='icrs')
        _, d2d, _ = det_sc.match_to_catalog_sky(true_sc)
        c_error = np.count_nonzero(d2d.degree < match_dist)
        error.append(c_error)
    error = np.array(error)
    return error.mean(), error.std() / np.sqrt(n_try - 1)

def extract_stats_from_df(df_name):
    import numpy as np
    import pandas as pd
    
    df = pd.read_csv(df_name)
    
    stat_df = {}
    stat_df['all'] = len(df)
    stat_df['all_tp'] = np.count_nonzero(df['status'] == 'tp')
    stat_df['all_fn'] = np.count_nonzero(df['status'] == 'fn')
    stat_df['fp'] = np.count_nonzero(df['status'] == 'fp')
    
    for cat in set(df['catalog']):
        if not (type(cat) == type('')):
            continue
        
        cur_df = df[df['catalog'] == cat]
        cur_df.index = np.arange(len(cur_df))
            
        stat_df[cat+'_tp'] = np.count_nonzero(cur_df['status'] == 'tp')
        stat_df[cat+'_fn'] = np.count_nonzero(cur_df['status'] == 'fn')
        stat_df[cat+'_recall'] = stat_df[cat+'_tp'] / len(cur_df)
    
    return stat_df

def stats_by_epoch(dir_name):
    import os
    import pandas as pd
    import numpy as np
    from DS_data_transformation import get_prm
    
    res_df = []
    
    files = next(os.walk(dir_name))[-1]
    for file in files:
        stat_line = extract_stats_from_df(os.path.join(dir_name, file))
        ep = int(get_prm('ep', file))
        res_df.append(pd.DataFrame(stat_line, index=[ep]))
        
    return pd.concat(res_df)


def plot_stats_ep(stats_df, text='', text_coords=[0,0]):
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    
    stats_df = stats_df.sort_index()
    cats_colors = {'planck_z' : 'b', 'planck_no_z' : 'g', 'mcxcwp' : 'r', 'actwp' : 'c'}
    

    fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    for cat in cats_colors:
        line, = ax[0].plot(stats_df.index, stats_df[cat+'_recall'], c=cats_colors[cat])
        line.set_label(cat)
    
    line, = ax[1].plot(stats_df.index, stats_df['fp'], c='k')
    line.set_label('fp')
    
    ax[0].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    ax[0].set_yticks(np.arange(0, 1.1, 0.2))
    ax[1].set_yticks(np.arange(0, max(stats_df['fp']), 50), minor=True)
    ax[1].set_yticks(np.arange(0, max(stats_df['fp']), 100))
    
    for i in range(2):
        ax[i].legend()
        ax[i].set_xticks(stats_df.index[4::5])
        ax[i].set_xticks(stats_df.index, minor=True)
        ax[i].grid(True, which='major')
        ax[i].grid(True, which='minor', alpha=0.2)
    
    ax[1].set_xlabel('epochs')
    ax[0].set_ylabel('recall')
    ax[1].set_ylabel('fp')
    ax[1].text(*text_coords, text, c='r', size=20)
