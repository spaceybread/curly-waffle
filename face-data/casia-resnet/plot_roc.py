from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import sys

rad_df = pd.read_csv('radial_classifier.csv')
gaus_df = pd.read_csv('gaussian_lattice.csv')
e8_df = pd.read_csv('e8_lattice.csv')
gaus_df = gaus_df.sort_values(by='TMR')
e8_df = e8_df.sort_values(by='TMR')

tmr_r = rad_df['TMR'].to_numpy()
fmr_r = rad_df['FMR'].to_numpy()

tmr_g = gaus_df['TMR'].to_numpy()
fmr_g = gaus_df['FMR'].to_numpy()

tmr_e = e8_df['TMR'].to_numpy()
fmr_e = e8_df['FMR'].to_numpy()


plt.figure(figsize=(8, 6))
plt.plot(fmr_r, tmr_r, color='darkorange', lw=2)
plt.plot(fmr_g, tmr_g, color='blue', lw=2)
plt.plot(fmr_e, tmr_e, color='green', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FMR')
plt.ylabel('TMR')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


