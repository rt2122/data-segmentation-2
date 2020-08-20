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
