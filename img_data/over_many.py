from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import sys

def get_data(npz_file = 'accum_data_dist.npy'):
    return np.load(npz_file, allow_pickle=True).item()


def bin_search(data):
    DELTA = 2
    hi, lo = DELTA, 0
    ALPHA = 2/3
    
    # lower bounded TMR
    # x/y result: coefficient [TMR, FMR]
    # 2/3 result: 0.940940409898758 [0.6666846603688709, 0.0030409356725146198]
    # 4/5 result: 1.0037665963172915 [0.8000089968511022, 0.00760233918128655]
    # 6/7 result: 1.067300617694855 [0.8571479982006298, 0.01537561853351327]
    
    keys = list(data.keys())
    res = {}
    
    for _ in range(64):
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

ma, val = bin_search(get_data())
print(val)

for x in ma:
    print(x, ma[x])
