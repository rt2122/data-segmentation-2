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