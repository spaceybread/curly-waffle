from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import sys

T = 10
R = 8

def first_load(npz_file, text_file):
    npz = np.load(npz_file)
    print(npz.shape)
    npz = npz.squeeze(1)
    print(npz.shape)
    ids = [x.strip().split('/')[-3] for x in open(text_file)]

    if (len(ids) != npz.shape[0]): print("Size Mismatch")
    
    grp = defaultdict(list)
    for s, v in zip(ids, npz): grp[s].append(v)

    rem = []
    for k in grp.keys():
    # replacement strat
        if len(grp[k]) < T + R: rem.append(k)
    
    for r in rem: grp.pop(r)
    np.save('mapped_data.npy', grp)
    

def test_set(npz_file):
    data = np.load(npz_file, allow_pickle=True).item()
    all_keys = list(data.keys())
    ma = {}
    mb = {}
    
    for x in data.keys():
        try:
            booga_boo = random.sample(data[x], k = R + T)
        except:
            print(len(data[x]), R, T)
            continue
        reg_vec = booga_boo[:R]
        in_set = booga_boo[R:]
        out_set = []
        out_key = x
        for i in range(T):
            while out_key == x:
                out_key = random.choices(all_keys, k = 1)[0]
            out_set.append(random.choices(data[out_key], k = 1)[0])
        
        center = np.mean(reg_vec, axis=0)
        distances = [np.linalg.norm(ve - center) for ve in reg_vec]
        init_rad = np.max(distances)
        
        in_dist = [np.linalg.norm(ve - center) for ve in in_set]
        out_dist = [np.linalg.norm(ve - center) for ve in out_set]
        
        ma[x] = (init_rad, in_dist, out_dist)
        mb[x] = (center, init_rad, in_set, out_set)
        # print(ma[x][0], ma[x][1])
    
    np.save('accum_data_dist.npy', ma)
    np.save('accum_data_for_lattice.npy', mb)

def load_accum(npz_file):
    data = np.load(npz_file, allow_pickle=True).item()
    
    # example
    print(len(data.keys()))
    print(list(data.keys())[2])
    print(data[list(data.keys())[2]][0])
    print(data[list(data.keys())[2]][1])
    print(data[list(data.keys())[2]][2])
    
    
first_load(sys.argv[1], sys.argv[2])
test_set('mapped_data.npy')
load_accum('accum_data_dist.npy')
