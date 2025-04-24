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

for PERSON in tqdm(range(1000)):
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

    for i in range(100):
        
    
    #print(r)
    
    # print(test_class)
    # print(tv)
    

        if start == -1: start = RC.alpha
#    if start1 == -1: start1 = RC1.get_radius()
    
        TMR = sum([1 if RC.resolve(tv.iloc[i].to_numpy()) else 0 for i in range(SAMPLES)])
        FMR = sum([1 if RC.resolve(fv.iloc[i].to_numpy()) else 0 for i in range(SAMPLES)])
        TMR, FMR = TMR / SAMPLES, FMR / SAMPLES
    
        if FMR > delta: RC.pull_alpha()
        else: RC.push_alpha()
    
        results.append((RC.alpha, TMR, FMR))
        final = RC.alpha
    
    
#    TMR = sum([1 if RC1.resolve(tv.iloc[i].to_numpy()) else 0 for i in range(SAMPLES)])
#    FMR = sum([1 if RC1.resolve(fv.iloc[i].to_numpy()) else 0 for i in range(SAMPLES)])
#    TMR, FMR = TMR / SAMPLES, FMR / SAMPLES
#
#    if FMR > delta: RC1.set_opt_rad(-1)
#    else: RC1.set_opt_rad(1)
#
#    results1.append((RC1.get_radius(), TMR, FMR))
#    final1 = RC1.get_radius()

    alphas.append(final)


print(alphas)
rounded_values = [round(v, 2) for v in alphas]
freq = Counter(rounded_values)

sorted_items = sorted(freq.items())
x_vals = [item[0] for item in sorted_items]
y_vals = [item[1] for item in sorted_items]

plt.figure(figsize=(10, 6))
plt.bar(x_vals, y_vals, width=0.01)
plt.xlabel('Rounded Value (2 decimal places)')
plt.ylabel('Frequency')
plt.title('Frequency of Rounded Float Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
