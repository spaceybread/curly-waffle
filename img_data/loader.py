import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import sys

MIN_IMG_TH = 10

def load_data(npz_file, info_file):
    npz = np.load(npz_file)
    # print(npz)
    inf = [[x.strip().split('/')[-2], x.strip().split('/')[-1]] for x in open(info_file)]
    daf = pd.DataFrame(npz)
    daf[['name', 'id']] = inf
    
    ids = daf.groupby('name').apply(lambda x: list(x.index), include_groups=False)
    ids = ids[ids.apply(lambda x: len(x) >= MIN_IMG_TH)]
    # print(f"data loaded. {daf.shape} entries. {ids.max()}")

    return daf, ids

def sample_ids(ids, t):
    # random class
    grp = ids.sample(1)
    grp_name = grp.index[0]
    # print(grp_name)
    
    # get t registration vectors
    reg_vals = np.random.choice(grp.iloc[0], t, replace=False)
    
    # + 1 from same group, candidate
    plus1 = np.random.choice(grp.iloc[0], 1)
    
    # get 1 from another group
    others = ids.drop(grp_name)
    other = np.random.choice(np.hstack(others.values), 1)
    
    # print(type(reg_vals), type(plus1), type(other))
    # print(reg_vals)
    # print(plus1, other)
    return reg_vals, plus1, other

def get_vectors(reg_vals, plus1, other, daf):
    reg_vectors = daf.loc[reg_vals].iloc[:, :-2]
    p1_vector = daf.loc[plus1].iloc[:, :-2]
    ot_vector = daf.loc[other].iloc[:, :-2]
    
    return reg_vectors, p1_vector, ot_vector

daf, ids = load_data(sys.argv[1], sys.argv[2])
reg_ids, p1_id, ot_id = sample_ids(ids, 6)
get_vectors(reg_ids, p1_id, ot_id, daf)

