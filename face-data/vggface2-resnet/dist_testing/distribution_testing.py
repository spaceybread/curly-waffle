from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import sys

T = 10
R = 8

import numpy as np
import polars as pl

def first_load(npz_file, text_file, output_parquet='keyed_data.parquet'):
    # Load feature vectors
    npz = np.load(npz_file)
    
    # Extract IDs from text file
    with open(text_file) as f:
        ids = [line.strip().split('/')[-2] for line in f]
    
    # Sanity check
    if len(ids) != npz.shape[0]:
        print(f"Size Mismatch: {len(ids)} ids vs {npz.shape[0]} vectors")
        return
    
    # Prepare rows for Polars
    data = []
    for key, vec in zip(ids, npz):
        data.append([key] + vec.tolist())

    # Build column names
    dim = npz.shape[1]
    col_names = ['KEY'] + [f'dim_{i}' for i in range(dim)]

    # Create Polars DataFrame
    df = pl.DataFrame(data, schema=col_names)

    # Save as Parquet
#    print(df.shape)
    df.write_parquet(output_parquet)
    print(f"Saved Polars DataFrame with shape {df.shape} to {output_parquet}")
    
first_load(sys.argv[1], sys.argv[2])

