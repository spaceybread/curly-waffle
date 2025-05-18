from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import sys

def get_data(npz_file = 'accum_data_dist.npy'):
    return np.load(npz_file, allow_pickle=True).item()


def bin_search(data, ALPHA = 2/3):
    DELTA = 2
    hi, lo = DELTA, 0
     
    # lower bounded TMR
    # x/y result: coefficient [TMR, FMR]
    # 2/3 result: 0.940940409898758 [0.6666846603688709, 0.0030409356725146198]
    # 4/5 result: 1.0037665963172915 [0.8000089968511022, 0.00760233918128655]
    # 6/7 result: 1.067300617694855 [0.8571479982006298, 0.01537561853351327]
    
    #updated dataset lower bounded TMR
    # TMR = x/y result: coefficient
    # 2/3 result: 0.9339694674126804
    # 4/5 result: 1.0276660318486392
    # 6/7 result: 1.085552275646478
    
    keys = list(data.keys())
    res = {}
    
    for _ in range(32):
        tchk, fchk = 0, 0
        tks, fks = 0, 0
        for key in keys:
            coeff = (hi + lo) / 2
            rad = data[key][0] * coeff
        
            tchk += sum([1 if val < rad else 0 for val in data[key][1]])
            tks += len(data[key][1])
            fchk += sum([1 if val < rad else 0 for val in data[key][2]])
            fks += len(data[key][2])
        
        tmr, fmr = tchk / tks, fchk / fks
        res[coeff] = [tmr, fmr]
        
        # do lower bounded TMR not upper bounded FMR
        if tmr > ALPHA: hi = coeff
        else: lo = coeff

    return res, coeff

#ma, val = bin_search(get_data())
#print(val, ma[val])
#
#exit()

DATA = get_data()
TMR, FMR = [], []
COEFFS = []
for i in range(1, 102, 1):
    resdb, idx = bin_search(DATA, ALPHA = i / 100)
    COEFFS.append(idx)
    TMR.append(resdb[idx][0])
    FMR.append(resdb[idx][1])

plt.figure(figsize=(8, 6))
plt.plot(FMR, TMR, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FMR')
plt.ylabel('TMR')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


