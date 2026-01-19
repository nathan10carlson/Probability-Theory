import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu0, mu1 = 110, 150
sig0, sig1 = 60, 100
n_values = [5, 15, 25]
t = np.linspace(mu0, mu1, 200)

# Colors for specificity and sensitivity
color_spec = '#8338ec'  # purple
color_sens = '#2a9d8f'  # green


# solid, dotted, extremely dotted
linestyles = ['-', ':', (0, (1, 5))]

plt.figure(figsize=(8.5, 5))

## Chat GPT helped with differernitating the lines
for i, n in enumerate(n_values):
    # Compute specificity and sensitivity
    spec = norm.cdf((t - mu0) / (sig0 / np.sqrt(n)))
    sens = 1 - norm.cdf((t - mu1) / (sig1 / np.sqrt(n)))

    # Plot specificity (purple)
    plt.plot(t, spec, color=color_spec, linestyle=linestyles[i], linewidth=2.0,
            markevery=15, markersize=5,
             label=f'Specificity (n={n})')

    # Plot sensitivity (green)
    plt.plot(t, sens, color=color_sens, linestyle=linestyles[i], linewidth=2.0,
             fillstyle='none', markevery=15, markersize=5,
             label=f'Sensitivity (n={n})', alpha=0.75)

# Labels and styling
plt.xlabel('Threshold $t$', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.grid()
plt.title('Sensitivity vs Specificity across Thresholds', fontsize=13, weight='bold')
plt.legend(frameon=True, shadow=False, fontsize=9, loc='lower right', ncol=2)
plt.ylim(-0.05, 1.05)
plt.xlim(mu0, mu1)
plt.tight_layout()
plt.show()