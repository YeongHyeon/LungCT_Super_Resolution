import os, glob
import scipy.misc
import numpy as np

list_npy = glob.glob(os.path.join("LungCT-Diagnosis", "*", "*.npy"))

list_tr = list_npy[:int(len(list_npy)*0.8)]
list_te = list_npy[int(len(list_npy)*0.8):]

try: os.mkdir("./dataset")
except: pass
try: os.mkdir("./dataset/low_tr")
except: pass
try: os.mkdir("./dataset/high_tr")
except: pass
try: os.mkdir("./dataset/low_te")
except: pass
try: os.mkdir("./dataset/high_te")
except: pass

for idx, ltr in enumerate(list_tr):
    origin = np.load(ltr)
    low = scipy.misc.imresize(origin, (int(origin.shape[0]/2), int(origin.shape[1]/2)))
    low = scipy.misc.imresize(low, (int(origin.shape[0]), int(origin.shape[1])), 'bilinear')
    np.save("./dataset/low_tr/%d" %(idx), low)
    np.save("./dataset/high_tr/%d" %(idx), origin)

for idx, lte in enumerate(list_te):
    origin = np.load(lte)
    low = scipy.misc.imresize(origin, (int(origin.shape[0]/4), int(origin.shape[1]/4)))
    low = scipy.misc.imresize(low, (int(origin.shape[0]), int(origin.shape[1])), 'bilinear')
    np.save("./dataset/low_te/%d" %(idx), low)
    np.save("./dataset/high_te/%d" %(idx), origin)
