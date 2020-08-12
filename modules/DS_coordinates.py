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


def sweep_to_dict(name):
    if name.endswith('.fits'):
        name = name[:-5]
    if name.startswith('sweep-'):
        name = name[len('sweep-'):]
    name = name.replace('p', '-')
    name = name.replace('m', '-')
    name = name.split('-')
    name = [float(k) for k in name]
    
    sweep_dict = {'RA1' : name[0], 'DEC1' : name[1], 'RA2' : name[2], 'DEC2' : name[3], 
                 'RA' : (name[0] + name[2]) / 2, 'DEC' : (name[1] + name[3]) / 2}
    
    return sweep_dict


def coords_in_sweep(sweepname, coords):
    import numpy as np
    sweep_dict = sweep_to_dict(sweepname)
    
    ra = coords[0]
    dec = coords[1]
    
    return np.logical_and(np.logical_and(np.logical_and(ra <= sweep_dict['RA2'], 
        ra >= sweep_dict['RA1']), dec <= sweep_dict['DEC2']), dec >= sweep_dict['DEC1'])
        
    
    
    
