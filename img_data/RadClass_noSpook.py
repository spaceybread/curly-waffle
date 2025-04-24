import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import sys

class RadialClassifier:
    def __init__(self, reg):
        self.center = reg.mean(axis=0)
        self.max_dist = self.get_def_rad(reg, mi = 0)
        
        self.hi, self.lo = 2, 1/16
        self.alpha = (self.hi * self.lo) ** 0.5
        
        self.radius = self.max_dist * self.alpha
    
    def push_alpha(self):
        print("bef:", self.alpha)
        self.lo = self.alpha
        self.alpha = (self.hi * self.lo) ** 0.5
        self.adj_rad()
        print("aft:", self.alpha)
    
    def pull_alpha(self):
        print("bef:", self.alpha)
        self.hi = self.alpha
        self.alpha = (self.hi * self.lo) ** 0.5
        self.adj_rad()
        print("aft:", self.alpha)
        
    def adj_rad(self):
        self.radius = self.max_dist * self.alpha
        
    def get_def_rad(self, reg, mi = 0):
        return max([np.linalg.norm(self.center - x) for x in reg])
        
    def resolve(self, candidate):
        return np.linalg.norm(self.center - candidate) <= self.radius
    
    def get_radius(self):
        return self.radius
    
    def set_radius(self, val):
        self.radius = val
