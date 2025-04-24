import matplotlib.pyplot as plt
from RadClass_noSpook import *
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
results1 = []

delta = 0.1
start = -1
final = -1

start1 = -1
final1 = -1
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
    
    if FMR > delta:
        RC.pull_alpha()
    else:
        RC.push_alpha()
    
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

results.sort()
results1.sort()
# print(results)
print(start, final)
print(len(results))



radius = [x[0] for x in results]
print(*radius)
tmr = [x[1] for x in results]
fmr = [x[2] for x in results]
plt.plot(radius, tmr, 'g-', label="TMR")
plt.plot(radius, fmr, 'r-', label="FMR")

#radius1 = [x[0] for x in results1]
#tmr1 = [x[1] for x in results1]
#fmr1 = [x[2] for x in results1]
#plt.plot(radius1, tmr1, 'g.', label="TMR - MIN")
#plt.plot(radius1, fmr1, 'r.', label="FMR - MIN")


plt.xlabel("Alpha")
plt.ylabel("Rate")
plt.title("TMR vs. FMR @ Radius")
plt.legend()


plt.show()
