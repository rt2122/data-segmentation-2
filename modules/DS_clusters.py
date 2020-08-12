class Cluster:
    
    def __init__(self, ra, dec, name):
        self.ra = ra
        self.dec = dec
        self.name = name

Virgo = Cluster(12 * 15 + 27 / 60, 
        12 + 43 / 60, 'Virgo Cluster')
Abell2029 = Cluster(15 * 15 + 10 / 60 + 56 / 3600, 
        5 + 44 / 60 + 41 / 3600, 'Abell 2029')
Abell1689= Cluster(13 * 15 + 11 / 60 + 34 / 3600, 
        -(1 + 21 / 60 + 56 / 3600) , 'Abell 1689')
