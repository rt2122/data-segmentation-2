val_pix = [10, 39, 42]
test_pix = [7]
train_pix = [i for i in range(48) if not (i in val_pix) and not (i in test_pix)]
planck_side = 2048
