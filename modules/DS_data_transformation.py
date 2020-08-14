def fits2df(fitsname):
    from astropy.io import fits
    from astropy.table import Table
    
    df = None
    with fits.open(fitsname) as hdul:
        tbl = Table(hdul[1].data)
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        df = tbl[names].to_pandas()
    return df

def pic2fits(pic, wcs, fitsname):
    from astropy.io import fits

    hdul = fits.HDUList([fits.PrimaryHDU(), 
        fits.ImageHDU(np.stack([pic[:,:,i] for i in range(pic.shape[-1])]), 
                     header=wcs.to_header())])
    hdul.writeto(fitsname)

def show_pic(pic, projection=None, label = 'label', figsize=(10, 10), vmin=0, vmax=1, 
        slices=None):
    from matplotlib import pyplot as plt
    fig= plt.figure(figsize=figsize)
    #ax= fig.add_axes([0.1,0.1,0.8,0.8], projection=projection, slices=slices)
    ax = plt.subplot(projection=projection, slices=slices)
    plt.xlabel(label)
    im = ax.imshow(pic, cmap=plt.get_cmap('viridis'),
            interpolation='nearest', vmin=vmin, vmax=vmax)


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

