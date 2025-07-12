import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.linspace(-1000, 1000, 5000)  
K = 50                            
R1 = 2
R2 = R1 * np.sqrt(2)
epsilon = 1e-10

sd_min_values = np.linspace(10, 50, 5)  # 15, 25, 35, 45, 55
fig, axes = plt.subplots(nrows=5, figsize=(12, 15), sharex=True)

for ax, sd_min in zip(axes, sd_min_values):
    x_i = np.random.uniform(-800, 800, K)  
    c = np.random.uniform(0.95, 1.05, K)   
    sd_i = sd_min * c             

    mix = np.zeros_like(x)
    for i in range(K):
        y = (1 / (sd_i[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - x_i[i]) / sd_i[i])**2)
        mix += y

    mix_interp = interp1d(x, mix, kind='linear', bounds_error=False, fill_value=0)

    x_plus_R1 = x + R1
    x_plus_R2 = x + R2
    x_min_R1 = x - R1
    x_min_R2 = x - R2

    mix_R1 = mix_interp(x_plus_R1)
    mix_R2 = mix_interp(x_plus_R2)
    mmix_R1 = mix_interp(x_min_R1)
    mmix_R2 = mix_interp(x_min_R2)

    ratio1 = mix_R2 / (mix_R1 + epsilon)
    ratio2 = mmix_R2 / (mmix_R1 + epsilon)

    ratio1[ratio1 < 0.99] = 0.99
    ratio2[ratio2 < 0.99] = 0.99

    ax.plot(x, ratio1, color='purple', label="FMR(x + R2) / FMR(x + R1)")
    ax.plot(x, ratio2, color='teal', label="FMR(x - R2) / FMR(x - R1)")
    ax.set_ylabel("Ratio")
    ax.set_title(f"sd_min = {sd_min:.0f}, K = {K}")
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("x")

plt.tight_layout()
plt.show()
