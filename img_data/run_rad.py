import matplotlib.pyplot as plt
from RadialClassifier import *
from sklearn import metrics
from loader import *
import pandas as pd
import numpy as np
import sys

SAMPLES = 100
T = 10
daf, ids = load_data(sys.argv[1], sys.argv[2])

test_class = sample_group_name(ids)
t_l, f_l = [], []

for i in range(SAMPLES):
    t, f = sample_tp(ids, test_class)
    t_l.append(t)
    f_l.append(f)

results = []

for i in range(100):
    reg_vals, _, _ = sample_ids_from_group(ids, test_class, T)
    r, tv, fv = get_vectors(reg_vals, t_l, f_l, daf)
    r = np.array([r.iloc[i].to_numpy() for i in range(T)])
    print(r)
    
    # print(test_class)
    # print(tv)
    RC = RadialClassifier(r)
    
    TMR = sum([1 if RC.resolve(tv.iloc[i].to_numpy()) else 0 for i in range(SAMPLES)])
    FMR = sum([1 if RC.resolve(fv.iloc[i].to_numpy()) else 0 for i in range(SAMPLES)])
    TMR, FMR = TMR / SAMPLES, FMR / SAMPLES
    
    results.append((RC.get_radius(), TMR, FMR))

results.sort()
print(results)


radius = [x[0] for x in results]
tmr = [x[1] for x in results]
fmr = [x[2] for x in results]
plt.plot(radius, tmr, 'g-', label="TMR")
plt.plot(radius, fmr, 'r-', label="FMR")

plt.xlabel("Radius")
plt.ylabel("Rate")
plt.title("TMR vs. FMR @ Radius")
plt.legend()


plt.show()
