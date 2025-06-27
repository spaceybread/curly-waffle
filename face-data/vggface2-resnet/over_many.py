from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import sys

def get_data(npz_file = 'accum_data_dist.npy'):
    return np.load(npz_file, allow_pickle=True).item()


def bin_search(data, ALPHA = 6/7):
    DELTA = 2
    hi, lo = DELTA, 0
    
    #updated dataset lower bounded TMR
    # TMR = x/y result: coefficient
    # 2/3, FMR = 0 result: 0.9367601573467254
    # 4/5, FMR = 0.0012 result: 1.0287885069847105
    # 6/7, FMR = 0.002 result: 1.0823243260383608
    
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

#ma, val = bin_search(get_data())
#print(ma)
#
#for k in ma.keys():
#    print(k, ma[k])
#print(val, ma[val])
#
#exit()

DATA = get_data()
TMR, FMR = [], []
COEFFS = []

res_ma = {
    "coeff": [], "TMR": [], "FMR": []
}


for i in range(1, 102, 1):
    resdb, idx = bin_search(DATA, ALPHA = i / 100)
    
    res_ma["coeff"].append(idx)
    res_ma["TMR"].append(resdb[idx][0])
    res_ma["FMR"].append(resdb[idx][1])
    
    pd.DataFrame.from_dict(res_ma, orient='columns').to_csv('radial_classifier.csv', index=False)

#plt.figure(figsize=(8, 6))
#plt.plot(FMR, TMR, color='darkorange', lw=2)
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('FMR')
#plt.ylabel('TMR')
#plt.title('Receiver Operating Characteristic')
#plt.legend(loc="lower right")
#plt.show()


