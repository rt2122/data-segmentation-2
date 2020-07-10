TARGET_RA1 = 154.76
TARGET_RA2 = 167.1
TARGET_DEC1 = 54.27
TARGET_DEC2 = 61.25

def line_in_field(line):
    import numpy as np

    def dot_in_field(ra, dec):
        return np.logical_and(np.logical_and(np.logical_and(
            ra <= TARGET_RA2, ra >= TARGET_RA1),
            dec <= TARGET_DEC2), dec >= TARGET_DEC1)

    return dot_in_field(line['RA'], line['DEC'])
