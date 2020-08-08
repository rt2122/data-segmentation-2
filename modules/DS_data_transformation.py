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
