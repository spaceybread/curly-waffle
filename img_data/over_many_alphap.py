from collections import Counter
import matplotlib.pyplot as plt
from RadClass_noSpook import *
from sklearn import metrics
from tqdm import tqdm
from loader import *
import pandas as pd
import numpy as np
import sys

SAMPLES = 100
T = 10
daf, ids = load_data(sys.argv[1], sys.argv[2])

alphas = []
Tf, Ff = [], []

for PERSON in tqdm(range(100)):
    test_class = sample_group_name(ids)
    t_l, f_l = [], []

    for i in range(SAMPLES):
        t, f = sample_tp(ids, test_class)
        t_l.append(t)
        f_l.append(f)

    results = []


    delta = 0.1
    start = -1
    final = -1

    reg_vals, _, _ = sample_ids_from_group(ids, test_class, T)
    r, tv, fv = get_vectors(reg_vals, t_l, f_l, daf)
    r = np.array([r.iloc[i].to_numpy() for i in range(T)])
    RC = RadialClassifier(r)

    RC.alpha = 1.76
    old = RC.radius
    RC.adj_rad()
    print(RC.alpha, old, "->", RC.radius)
    
    TMR = sum([1 if RC.resolve(tv.iloc[i].to_numpy()) else 0 for i in range(SAMPLES)])
    FMR = sum([1 if RC.resolve(fv.iloc[i].to_numpy()) else 0 for i in range(SAMPLES)])
    TMR, FMR = TMR / SAMPLES, FMR / SAMPLES
    
    Tf.append(TMR)
    Ff.append(FMR)


print(max(Tf), min(Tf), sum(Tf) / len(Tf))
print(max(Ff), min(Ff), sum(Ff) / len(Ff))
