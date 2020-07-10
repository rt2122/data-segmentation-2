TARGET_RA1 = 154.76
TARGET_RA2 = 167.1
TARGET_DEC1 = 54.27
TARGET_DEC2 = 61.25

def line_in_field(line):

    def dot_in_field(ra, dec):
        if ra <= TARGET_RA2 and ra >= TARGET_RA1 and \
            dec <= TARGET_DEC2 and dec >= TARGET_DEC1:
            return True
        return False

    return dot_in_field(line['RA'], line['DEC'])
