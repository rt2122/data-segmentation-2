val_pix = [10, 39, 42]
test_pix = [7]
train_pix = [i for i in range(48) if not (i in val_pix) and not (i in test_pix)]
planck_side = 2048
min_rad_64 = 0.62

def gen_matr(ra, dec, radius=0.84, size=64, fin_nside=2048):
    from DS_healpix_fragmentation import find_biggest_pixel, one_pixel_fragmentation,\
        draw_proper_circle, matr2dict
    import numpy as np
    
    big_nside, big_ipix = find_biggest_pixel(ra, dec, radius)
    depth = int(np.log2(fin_nside // big_nside))
    big_matr = one_pixel_fragmentation(big_nside, big_ipix, depth)
    big_dict = matr2dict(big_matr)
    
    
    circle_coords = draw_proper_circle(ra, dec, radius, fin_nside, big_dict, 
                                       big_matr.shape, coords_mode=True)
    mins = [None, None]
    maxs = [None, None]
    for i in range(2):
        dif = circle_coords[:,i].min() - circle_coords[:,i].max() + size
        mins[i] = circle_coords[:,i].min() - dif
        maxs[i] = circle_coords[:,i].max() + dif
        if maxs[i] - mins[i] != size:
            maxs[i] -= maxs[i] - mins[i] - size
    
   
    return big_matr[mins[0]:maxs[0],mins[1]:maxs[1]]

def draw_pic(matr, dirname='/home/rt2122/Data/Planck/normalized/'):
    import os
    import numpy as np
    
    files = sorted(next(os.walk(dirname))[-1])
    pic = np.zeros(list(matr.shape) + [len(files)])
    
    for i_f, file in enumerate(files):
        i_s = np.load(os.path.join(dirname, file))
        
        for x in range(pic.shape[0]):
            pic[x, :, i_f] = i_s[matr[x]]
    return pic

def draw_pic_with_mask(center, clusters, radius=0.84, size=64, fin_nside=2048, 
                       dirname='/home/rt2122/Data/Planck/normalized/', 
                      mask_radius=5/60):
    from DS_healpix_fragmentation import matr2dict, draw_proper_circle
    import numpy as np
    
    matr = gen_matr(center[0], center[1], radius, size, fin_nside)
    mdict = matr2dict(matr)
    
    pic = draw_pic(matr, dirname)
    
    mask = np.zeros(list(matr.shape) + [1], dtype=np.uint8)
    for ra, dec in clusters:
        mask = np.logical_or(mask, 
            draw_proper_circle(ra, dec, mask_radius, fin_nside, mdict, 
                              mask.shape, coords_mode=False))
    
    return pic, mask

def nearest_clusters(df, theta, phi, radius=2, galactic=True):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    sc_cen = None
    sc_cl = SkyCoord(ra=np.array(df['RA'])*u.degree, 
                     dec=np.array(df['DEC'])*u.degree, frame='icrs')
    if galactic:
        sc_cen = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
    else:
        sc_cen = SkyCoord(ra=theta*u.degree, dec=phi*u.degree, frame='icrs')
    return df[sc_cen.separation(sc_cl).degree < radius]

def pixels_with_clusters(clusters, big_pixels, nside, min_rad=0.62):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import pandas as pd
    import healpy as hp
    import numpy as np
    from DS_healpix_fragmentation import radec2pix

    
    df = pd.read_csv(clusters)
    pix2 = radec2pix(df['RA'], df['DEC'], 2)
    df = df[np.in1d(pix2, big_pixels)]
    df.index = np.arange(df.shape[0])
    
    small_pixels = set()
    sc = SkyCoord(ra=np.array(df['RA'])*u.degree, 
                  dec=np.array(df['DEC'])*u.degree, frame='icrs')
    vecs = hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree, 
                      lonlat=True)
    for i in range(df.shape[0]):
        small_pixels = small_pixels.union(hp.query_disc(nside, vecs[i], 
                                                np.radians(min_rad), nest=True))
    small_pixels = np.array(list(small_pixels))
    return small_pixels, df

def gen_data(clusters, big_pixels, batch_size, nside=2048, min_rad=0.62, search_nside=512,
        output=False):
    import healpy as hp
    import numpy as np
    from DS_healpix_fragmentation import radec2pix
    from DS_Planck_Unet import draw_pic_with_mask
    #from tqdm.notebook import tqdm
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    
    small_pixels, df = pixels_with_clusters(clusters, big_pixels, search_nside, min_rad)
    
    while True:
        ipix = np.random.choice(small_pixels, batch_size)
        theta, phi = hp.pix2ang(nside=search_nside, nest=True, ipix=ipix, lonlat=True)
        sc = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
        ra = sc.icrs.ra.degree
        dec = sc.icrs.dec.degree
        pics = []
        masks = []
        i = 0
        while i < batch_size:
            cl_list = nearest_clusters(df, theta[i], phi[i], galactic=True)
            cl_list = np.stack([cl_list['RA'], cl_list['DEC']]).T
            pic, mask = draw_pic_with_mask([ra[i], dec[i]], cl_list)
            if output:
                print(i, ipix[i], pic.shape, ra[i], dec[i])
            if not(pic.shape[0] == 64 and pic.shape[1] == 64):
                small_pixels = small_pixels[small_pixels != ipix[i]]
                ipix[i] = np.random.choice(small_pixels)
                theta[i], phi[i] = hp.pix2ang(nside=search_nside, nest=True, 
                        ipix=ipix[i], lonlat=True)
                sc_cur = SkyCoord(l=theta[i]*u.degree, b=phi[i]*u.degree, frame='galactic')
                ra[i] = sc_cur.icrs.ra.degree
                dec[i] = sc_cur.icrs.dec.degree
            else:
                pics.append(pic)
                masks.append(mask)
                i += 1
        yield np.stack(pics), np.stack(masks)


def iou(y_pred, y_true):
    from tensorflow.keras import backend as K
    iou_sum = 0
    for i in range(y_true.shape[-1]):
        inters = K.sum(y_pred[..., i] * y_true[..., i])
        union = K.sum((y_pred[..., i] + y_true[..., i])) - inters
        iou_sum += inters / union
    return iou_sum

def dice(y_pred, y_true, eps=0.1):
    from tensorflow.keras import backend as K
    dice_sum = 0
    for i in range(y_true.shape[-1]):
        inters = K.sum(y_pred[..., i] * y_true[..., i])
        union = K.sum((y_pred[..., i] + y_true[..., i])) - inters
        dice_sum += K.mean((2 * inters + eps) / (union + eps))
    return dice_sum


def unet_planck(input_size = (64,64,6), filters=16, blocks=5, output_layers=1, weights=None): 
    import numpy as np 
    from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
    from tensorflow.keras import Input
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.activations import relu, sigmoid
    from tensorflow.keras.layers import UpSampling2D
    from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy,\
            sparse_categorical_crossentropy
    from tensorflow.keras.layers import BatchNormalization, Dropout
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model

    encoder = []
    inputs = Input(input_size)
    prev = inputs
    for i in range(blocks):
        cur = Conv2D(filters=filters, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(prev)
        #cur = BatchNormalization()(cur)
        cur = Activation(relu)(cur)
        cur = Dropout(0.2)(cur)

        cur = Conv2D(filters=filters, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(cur)
        #cur = BatchNormalization()(cur)
        cur = Dropout(0.2)(cur)
        cur = Activation(relu)(cur)

        encoder.append(cur)

        cur = MaxPooling2D(padding='valid')(cur)

        filters *= 2
        prev = cur
    for i in range(blocks - 1, -1, -1):
        cur = UpSampling2D()(prev)
        cur = Conv2D(filters=filters, kernel_size=3, padding='same')(cur)
        #cur = Dropout(0.2)(cur)
        cur = Activation(relu)(cur)
        cur = concatenate([cur, encoder[i]], axis=3)

        cur = Conv2D(filters=filters, kernel_size=3, padding='same')(cur)
        cur = Activation(relu)(cur)
        #cur = Dropout(0.2)(cur)
        #cur = Conv2D(filters=filters, kernel_size=3, padding='same')(cur)
        #cur = Activation(relu)(cur)

        prev = cur
        filters //= 2

    if not (weights is None):
        pt = Model(inputs=inputs, outputs=prev)
        pt.load_weights(weights)

    prev = Conv2D(output_layers, kernel_size=3, padding='same')(prev)
    prev = Activation(sigmoid)(prev)
    
    model = Model(inputs=inputs, outputs=prev)
    model.compile(optimizer = Adam(lr = 1e-4), loss = binary_crossentropy, metrics = ['accuracy', iou, dice])
    
    return model
