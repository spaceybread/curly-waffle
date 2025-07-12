import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Load data
rad_df = pd.read_csv('radial_classifier.csv')
gaus_df = pd.read_csv('gaussian_lattice.csv')

# Sort by TMR for better plotting
rad_df = rad_df.sort_values(by='TMR')
gaus_df = gaus_df.sort_values(by='TMR')

# Get all unique r_size values
r_sizes_rad = rad_df['r_size'].unique()
r_sizes_gaus = gaus_df['r_size'].unique()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot for radial data
for r in r_sizes_rad:
    sub_df = rad_df[rad_df['r_size'] == r]
    axes[0].plot(np.log(sub_df['FMR']) / np.log(2), sub_df['TMR'], label=f"r_size={r}")
#axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.5, 1.001])
axes[0].set_xlabel('FMR')
axes[0].set_ylabel('TMR')
axes[0].set_title('Radial Classifier ROC')
axes[0].legend(loc='lower right')

# Plot for gaussian data
for r in r_sizes_gaus:
    sub_df = gaus_df[gaus_df['r_size'] == r]
    axes[1].plot(np.log(sub_df['FMR']) / np.log(2), sub_df['TMR'], label=f"r_size={r}")
#axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#axes[1].set_xlim([0.0, 1.0])
axes[1].set_xlabel('FMR')
axes[1].set_title('Gaussian Lattice ROC')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()
