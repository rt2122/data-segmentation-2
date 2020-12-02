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

def get_prm(prm, s):
    import re
    return re.search(prm + '(\w+)_', s)[0][len(prm):]
