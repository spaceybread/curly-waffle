from tqdm import tqdm
import numpy as np
import subprocess


def get_data(npz_file = 'accum_data_for_lattice.npy'):
    return np.load(npz_file, allow_pickle=True).item()

def call_check(scale, a, b):
    a_str, b_str = vec_to_string(a), vec_to_string(b)
    
    EXEC = "./lattice_fuzzy"
    args = [EXEC, str(scale), a_str, b_str]
    result = subprocess.run(args, capture_output=True, text=True)
    
    match = result.stdout.strip()
    return match == "match!"

def vec_to_string(vec):
    vec_str = [str(v) for v in vec]
    return " ".join(vec_str)
    

def data_sanity_parsing(data):
    keys = list(data.keys())
    key = keys[0]

#    print(vec_to_string(data[key][3][0]))
    
    print(data[key][1])

    print(call_check(data[key][1], data[key][0], data[key][2][0]))
    
    
def bin_search(data, ALPHA = 2/3):
    DELTA = 10
    hi, lo = DELTA, 0
     
    keys = list(data.keys())
    res = {}
    
    for _ in tqdm(range(32)):
        tchk, fchk = 0, 0
        tks, fks = 0, 0
        for key in tqdm(keys):
        
            coeff = (hi + lo) / 2
            rad = data[key][0] * coeff
        
            tchk += sum([1 if call_check(rad, data[key][0], val) else 0 for val in data[key][2]])
            tks += len(data[key][2])
            fchk += sum([1 if call_check(rad, data[key][0], val) else 0 for val in data[key][3]])
            fks += len(data[key][3])
        
        tmr, fmr = tchk / tks, fchk / fks
        res[coeff] = [tmr, fmr]
        
        # do lower bounded TMR not upper bounded FMR
        if tmr > ALPHA: hi = coeff
        else: lo = coeff

    return res, coeff
    



scale = "1.0"
a = "1.0 " * 512
b = "1.0 " * 512


data = get_data()
out = bin_search(data)

for r in out:
    print(r, out[r])

