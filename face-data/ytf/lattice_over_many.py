from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from functools import partial
from tqdm import tqdm
import numpy as np
import pandas as pd
import subprocess

def get_data(npz_file = 'accum_data_for_lattice.npy'):
    return np.load(npz_file, allow_pickle=True).item()

def call_check(scale, a, b):
    a_str = vec_to_string(a)
    b_str = vec_to_string(b)
    
    EXEC = "./lattice_fuzzy"
    args = [EXEC, str(scale), a_str, b_str]
    result = subprocess.run(args, capture_output=True, text=True)
    
    match = result.stdout.strip()
    if match != "match!" and match != "no match!":
        print(result.stdout + "\n\n" + result.stderr)
        print(a)
        print(b)
        print(scale)
        quit()

    return match == "match!"

def vec_to_string(vec):
    vec_str = [str(v) for v in vec]
    return " ".join(vec_str)
    

def data_sanity_parsing(data):
    keys = list(data.keys())
    key = keys[0]
    
    cen = data[key][0]
    rad = data[key][1]

    
    task = partial(call_check, rad * 5, cen)
    tchk = 0
    fchk = 0
    with ProcessPoolExecutor() as exec:
        fut = [exec.submit(task, val) for val in t_v]
        for f in as_completed(fut):
            if f.result(): tchk += 1
    with ProcessPoolExecutor() as exec:
        fut = [exec.submit(task, val) for val in f_v]
        for f in as_completed(fut):
            if f.result(): tchk += 1
    
    print(tchk, fchk)
    
    
def bin_search(data, ALPHA = 2/3):
    DELTA = 10
    hi, lo = DELTA, 0
     
    keys = list(data.keys())
    res = {}
    exec = ProcessPoolExecutor()
    for _ in tqdm(range(16), desc="bin search"):
        tchk, fchk = 0, 0
        tks, fks = 0, 0
        coeff = (hi + lo) / 2
        
        if mem.get(coeff) != None:
            if mem.get(coeff)[0] > ALPHA: hi = coeff
            else: lo = coeff
            continue

        for key in tqdm(keys, desc="keys"):
            rad = data[key][1] * coeff
            t_v = data[key][2]
            f_v = data[key][3]
            cen = data[key][0]
            
            task = partial(call_check, rad, cen)
            
            t_results = exec.map(task, t_v)
            f_results = exec.map(task, f_v)

            tchk += sum(t_results)
            fchk += sum(f_results)

            tks += len(t_v)
            fks += len(f_v)
            
        tmr, fmr = tchk / tks, fchk / fks
        res[coeff] = [tmr, fmr]
        mem[coeff] = [tmr, fmr]
        print(coeff, tmr, fmr)
        # do lower bounded TMR not upper bounded FMR
        if tmr > ALPHA: hi = coeff
        else: lo = coeff
        
    exec.shutdown()
    return res, coeff





    


if __name__ == '__main__':
#    scale = "1.0"
#    a = "1.0 " * 512
#    b = "1.0 " * 512
#
#
#
##    data_sanity_parsing(data)
#
#    np.save('processed_lattice.npy', ma)
#
#    for r in ma.keys():
#        print(r, ma[r])

    res_ma = {
        "coeff": [], "TMR": [], "FMR": []
    }
    data = get_data()
    
    mem = {}

    for i in tqdm(range(5, 101, 5), desc="overall"):
        ma, val = bin_search(data, ALPHA = i / 100)
        for v in ma.keys():
            res_ma["coeff"].append(v)
            res_ma["TMR"].append(ma[v][0])
            res_ma["FMR"].append(ma[v][1])
        
        pd.DataFrame.from_dict(res_ma, orient='columns').to_csv('e8_lattice.csv', index=False)
