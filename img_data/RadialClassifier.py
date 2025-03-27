import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import sys

class RadialClassifier:
    def __init__(self, reg, rad = -1):
        self.center = reg.mean(axis=0)
        self.radius = rad if rad != -1 else self.get_def_rad(reg)
    
    def get_def_rad(self, reg):
        return max([np.linalg.norm(self.center - x) for x in reg])
        
    def resolve(self, candidate):
        return np.linalg.norm(self.center - candidate) < self.radius
