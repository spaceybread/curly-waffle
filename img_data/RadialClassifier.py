import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import sys

class RadialClassifier:
    def __init__(self, reg, rad = -1, mit = 0):
        self.center = reg.mean(axis=0)
        
        self.lo = self.get_def_rad(reg, mi = mit)
        self.hi = self.get_def_rad(reg, mi = mit) * 2
        self.radius = self.compute_opt_r()
        
    def get_def_rad(self, reg, mi = 0):
        # print([np.linalg.norm(self.center - x) for x in reg])
        if mi == 0: return max([np.linalg.norm(self.center - x) for x in reg])
        return min([np.linalg.norm(self.center - x) for x in reg])
    
    def compute_opt_r(self):
        return (self.hi + self.lo) / 2
        
    def set_opt_rad(self, M = 0):
        if M == 0: return self.compute_opt_r()
        
        if M == 1:
            self.lo = self.radius
            self.radius = self.compute_opt_r()
        
        if M == -1:
            self.hi = self.radius
            self.radius = self.compute_opt_r()
        
    def resolve(self, candidate):
        return np.linalg.norm(self.center - candidate) <= self.radius
    
    def get_radius(self):
        return self.radius
    
    def set_radius(self, val):
        self.radius = val
