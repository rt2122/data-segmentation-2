import sys
from gen_pred_ind import radec2pred_ind

args = None
with open(sys.argv[1], 'r') as f:
    args = f.read().split('\n')

args = list(filter(lambda x: len(x) > 0, args))

print("Generating pred_ind from scans: ", args[0])
print("Input file: ", args[1])
print("Output file: ", args[2])
t = 'fits'
if len(args) > 3:
    t = args[3]
print("Type of input file: ", t)
radec2pred_ind(*args)

