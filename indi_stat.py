import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1000, 1000, 5000)
K = 50
sd_min = 30
np.random.seed(314)

x_i = np.random.uniform(-800, 800, K)
c = np.random.uniform(1, 1.05, K)
sd_i = sd_min * c

plt.figure(figsize=(12, 6))

mix = np.zeros_like(x)

for i in range(K):
    y = (1 / (sd_i[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - x_i[i]) / sd_i[i])**2)
    mix += y
    plt.plot(x, y, linestyle='--', alpha=0.6)

plt.plot(x, mix, color='black', linewidth=2)

plt.title(f"Mixture of {K} Gaussians (sd_min = {sd_min})")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()

