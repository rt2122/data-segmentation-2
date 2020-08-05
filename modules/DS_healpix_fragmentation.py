def one_pixel_fragmentation(o_nside, o_pix, depth):
    import healpy as hp
    import numpy as np
    
    def recursive_fill(matr):
        if matr.shape[0] == 1:
            return

        mid = matr.shape[0] // 2
        np.left_shift(matr, 1, out=matr)
        matr[mid:, :] += 1

        np.left_shift(matr, 1, out=matr)
        matr[:, mid:] += 1

        for i in [0, mid]:
            for j in [0, mid]:
                recursive_fill(matr[i:i+mid, j:j+mid])
                
    m_len = 2 ** depth
    f_matr = np.full((m_len, m_len), o_pix)
    
    recursive_fill(f_matr)
    return f_matr


def find_biggest_pixel(ra, dec, radius):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    import numpy as np

    
    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    nside = 1
    vec = hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree,
                    lonlat=True)
    radius = np.radians(radius)
    
    while len(set(hp.query_disc(nside, vec, radius, inclusive=True, 
                                nest=True))) != 1 and\
        len(set(hp.query_disc(nside, vec, radius, inclusive=False, 
                                nest=True))) != 1:
        nside *= 2
        if nside > 2**14:
            break
    if nside > 1:
        nside = nside // 2
    return nside, hp.vec2pix(nside, *vec, nest=True)


def matr2dict(matr):
    d = {}
    for i in range(matr.shape[0]):
        for j in range(matr.shape[1]):
            d[matr[i, j]] = (i, j)
    return d

def radec2pix(ra, dec, nside):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp

    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    return hp.ang2pix(nside, sc.galactic.l.degree, sc.galactic.b.degree, 
                                  nest=True, lonlat=True)
    

def draw_circles(ra, dec, nside, shape, mdict, radius=None):
    import numpy as np
    from skimage.draw import circle

    if radius is None:
        radius = int(max(shape) / 20)
    pic = np.zeros(shape, dtype=np.uint8)
    pix = radec2pix(ra, dec, nside)
    coords = [mdict[p] for p in pix]
    for x, y in coords:
        pic[circle(x, y, radius, shape=shape)] = 1
    
    return pic

def draw_proper_circle(ra, dec, nside, shape, mdict, radius, mode='coords'):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    import numpy as np

    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    vec = hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree, lonlat=True)
    pix = hp.query_disc(nside, vec, np.radians(radius), nest=True, inclusive=True)
    coords = [mdict[p] for p in pix if p in mdict]
    if mode == 'coords':
        return np.array(coords)
    
    pic = np.zeros(shape, dtype=np.uint8)
    for x, y in coords:
        pic[x, y, 0] = 1
    return pic 

def nearest_power(n):
    k = 0
    isPower = True
    while n > 0:
        if n % 2 != 0 and n > 1:
            isPower = False
        n //= 2
        k += 1
    if isPower:
        k -= 1
    return 2**k

def zoom_to_circle(coords, matr):
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    xdif = xmax - xmin
    ydif = ymax - ymin
    
    map_size = nearest_power(max(xdif, ydif))
    xmax += map_size - xdif
    ymax += map_size - ydif
    
    return matr[xmin:xmax, ymin:ymax]

def draw_data(mdict, nside, shape, fits_name):
    import numpy as np
    from astropy.io import fits
    from tqdm.notebook import tqdm

    data_pic = np.zeros(shape)
    with fits.open(fits_name) as hdul:
        
        data = hdul[1].data
        flux = (data['FLUX_G'], data['FLUX_R'], data['FLUX_Z'])
        ra = data['RA']
        dec = data['DEC']
        pixels = radec2pix(ra, dec, nside)

        coords = []
        for pix in tqdm(pixels):
            if pix in mdict:
                coords.append(mdict[pix])

        for i in tqdm(range(len(coords))):
            x, y = coords[i]
            for k, ch in enumerate(flux):
                data_pic[x, y, k] = max(data_pic[x, y, k], ch[i])
        return data_pic

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
