def find_nearest_tile(ra, dec):
    from astropy.coordinates import SkyCoord 
    from astropy.wcs import WCS
    from astropy.io import fits
    from astropy import units as u
    import numpy as np

    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    line = None
    
    with fits.open('/home/rt2122/Data/fulldepth_neo4_index.fits') as hdul:
        sc1 = SkyCoord(ra=hdul[1].data['RA']*u.degree, 
                       dec=hdul[1].data['DEC']*u.degree)
        idx = np.argmin(sc.separation(sc1).degree)
        line = hdul[1].data[idx]
    
    w = WCS(naxis=2)
    w.wcs.cd = line['CD']
    w.wcs.cdelt = line['CDELT']
    w.wcs.crpix = line['CRPIX']
    w.wcs.crval = line['CRVAL']
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.lonpole = line['LONGPOLE']
    w.wcs.latpole = line['LATPOLE']
    w.wcs.set_pv([(0, 0, 0)])
    w.array_shape = (2048, 2048)
    return w

def custom_wcs(ra, dec):
    from astropy.io import fits
    from astropy.wcs import WCS

    cutout_url = 'https://www.legacysurvey.org/viewer/cutout.fits?ra={:.4f}&dec={:.4f}&layer=sdss2&pixscale=4.00'
    w = None
    with fits.open(cutout_url.format(ra, dec)) as hdul:
         w = WCS(hdul[0].header)
    w1 = WCS(naxis=2)
    w1.wcs.cd = w.wcs.cd[:2,:2]
    w1.wcs.cdelt = w.wcs.cdelt[:2]
    w1.wcs.crpix = [1024.5, 1024.5]
    w1.wcs.crval = w.wcs.crval[:2]
    w1.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w1.wcs.lonpole = w.wcs.lonpole
    w1.wcs.latpole = w.wcs.latpole
    w1.wcs.set_pv([(0, 0, 0)])
    w1.array_shape = [2048, 2048]
    return w1

def ra_dec2pixels(ra_center, dec_center, ra, dec, custom=True):
    import numpy as np

    if not custom:
        w = find_nearest_tile(ra_center, dec_center)
        return np.array(w.all_world2pix(ra, dec, 0))
    w = custom_wcs(ra_center, dec_center)
    return np.array(w.all_world2pix(ra, dec, np.zeros((ra.shape)), 0))

def draw_data(arr, channels_data, pixels, func=max):
    for i in range(pixels.shape[0]):
        x, y = int(pixels[i][0]), int(pixels[i][1])
        if x >= 0 and y >= 0 and \
            x < arr.shape[0] and y < arr.shape[1]:
            for j, ch in enumerate(channels_data):
                arr[x, y, j] = func(arr[x, y, j], ch[i])


def show_pic(pic, projection=None, label = 'label', figsize=(10, 10), vmin=0, vmax=1):
    from matplotlib import pyplot as plt
    fig= plt.figure(figsize=figsize)
    ax= fig.add_axes([0.1,0.1,0.8,0.8], projection=projection)
    plt.xlabel(label)
    im = ax.imshow(pic, cmap=plt.get_cmap('viridis'),
            interpolation='nearest', vmin=vmin, vmax=vmax)

def dist_from_center(wcs_proj):
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    pixels = np.array([[0, wcs_proj.wcs.crpix[0]], wcs_proj.wcs.crpix], dtype=int)
    pixels_t = wcs_proj.all_pix2world(pixels[:, 0], pixels[:, 1], 0)
    sc = SkyCoord(ra=pixels_t[0]*u.degree, dec=pixels_t[1]*u.degree, frame='icrs')
    return sc[0].separation(sc[1]).degree
