import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load both datasets
df1 = pd.read_csv("radial_classifier.csv")
df2 = pd.read_csv("gaussian_lattice.csv")

# Round TMR to avoid float noise
df1["TMR_rounded"] = df1["TMR"].round(3)
df2["TMR_rounded"] = df2["TMR"].round(3)

# Define target TMRs and assign a unique marker to each
target_tmrs = [round(x, 3) for x in [2/3, 4/5, 6/7]]
markers = ['s', '^', 'D']  # Circle, square, triangle, diamond

# Prepare figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

def plot_dataset(ax, df, title):
    for tmr, marker in zip(target_tmrs, markers):
        subset = df[df["TMR_rounded"] == tmr]
        if not subset.empty:
            # Plot original data
            ax.plot(subset["r_size"], subset["FMR"], marker=marker, linestyle='-', label=f"TMR ≈ {tmr}")
            
            # Fit line (1st-degree polynomial)
            coeffs = np.polyfit(subset["r_size"], subset["FMR"], 1)
            slope, intercept = coeffs
            
            # Generate line of best fit
            x_vals = np.linspace(subset["r_size"].min(), subset["r_size"].max(), 100)
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, linestyle='--', linewidth=1.5,
                    label=f"slope ≈ {slope:.5f}", alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("r_size")
    ax.set_ylabel("FMR")
    ax.grid(True)
    ax.legend()

# Plot both datasets
plot_dataset(ax1, df1, "Radial Classifier")
plot_dataset(ax2, df2, "Gaussian Classifier")

#plt.suptitle("FMR vs r_size with Linear Fit for Selected TMR Values")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
plt.show()
