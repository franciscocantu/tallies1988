import os
import sys
import random

dir_from = sys.argv[1]
dir_to = sys.argv[2]
N = int(sys.argv[3])

i = 0
while i < N:
    f = os.path.join(dir_from, random.choice(os.listdir(dir_from)))
    if 'copy' not in f:
        os.system('mv "' + f + '" ' + os.path.join(dir_to))
        i = i + 1
