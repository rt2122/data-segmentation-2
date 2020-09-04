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


def find_biggest_pixel(ra, dec, radius, root_nside=1, max_nside=32):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    import numpy as np

    nside = root_nside
    radius = np.radians(radius)
    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    theta = sc.galactic.l.degree
    phi = sc.galactic.b.degree
    vec = hp.ang2vec(theta=theta, phi=phi, lonlat=True)
    
    pixels = hp.query_disc(vec=vec, nside=nside, radius = radius, inclusive=False, 
                                nest=True)
    while len(pixels) <= 1:
        if nside == max_nside:
            break
        nside *= 2
        pixels = hp.query_disc(vec=vec, nside=nside, radius = radius, inclusive=False, 
                                    nest=True)
    if nside > 1:
        nside //= 2
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
def pix2radec(ipix, nside):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp

    theta, phi = hp.pix2ang(nside, ipix=ipix, nest=True, lonlat=True)

    sc = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
    return sc.icrs.ra.degree, sc.icrs.dec.degree     

def draw_circles_h(ra, dec, data, nside, mdict, shape, coef=0.02):
    import numpy as np
    from skimage.draw import circle

    coef = shape[0] * coef / max(data)
    pic = np.zeros(shape, dtype=np.uint8)
    pix = radec2pix(ra, dec, nside)
    coords = [mdict[p] for p in pix if p in mdict]
    for i in range(len(data)):
        x, y = coords[i]
        pic[circle(x, y, data[i] * coef, shape=shape)] = 1
    
    return pic

def draw_dots_h(ra, dec, data, nside, mdict, shape):
    import numpy as np

    pic = np.zeros(shape)
    pix = radec2pix(ra, dec, nside)
    coords = [mdict[p] for p in pix if p in mdict]
    if len(shape) == 2:
        for i in range(len(data)):
            x, y = coords[i]
            pic[x, y] = data[i]
    else:
        for i in range(len(data)):
            x, y = coords[i]
            pic[x, y, 0] = data[i]

    
    return pic

def draw_proper_circle(ra, dec, radius, nside, mdict, shape, coords_mode=True):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import healpy as hp
    import numpy as np

    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    vec = hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree, lonlat=True)
    pix = hp.query_disc(nside, vec, np.radians(radius), nest=True, inclusive=True)
    coords = [mdict[p] for p in pix if p in mdict]
    if coords_mode:
        return np.array(coords)
    
    pic = np.zeros(shape, dtype=np.uint8)
    if len(shape) == 2:
        for x, y in coords:
            pic[x, y] = 1
    else:
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

def zoom_to_circle(coords, matr, add_power=True):
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    if not add_power:
        return matr[xmin:xmax, ymin:ymax]
    xdif = xmax - xmin
    ydif = ymax - ymin
    
    map_size = nearest_power(max(xdif, ydif))
    xdif = map_size - xdif
    ydif = map_size - ydif
    xmin -= xdif // 2 + xdif % 2
    ymin -= ydif // 2 + ydif % 2
    xmax += xdif // 2
    ymax += ydif // 2
    
    return matr[xmin:xmax, ymin:ymax]
'''
def draw_data_fits(mdict, nside, shape, fits_name):
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
'''
def show_dict_mollview(mdict, nside):
    import healpy as hp
    import numpy as np

    a = np.zeros((hp.nside2npix(nside)))
    for pix in mdict:
        a[pix] = 1
    hp.mollview(a, nest=True)

