import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters from earlier
mu0, mu1 = 110, 150
sig0, sig1 = 60, 100
t, n = 125, 25

# Compute specificity and sensitivity
spec = norm.cdf((t - mu0) / (sig0 / np.sqrt(n)))        # P(Zn < t | Y=0)
sens = 1 - norm.cdf((t - mu1) / (sig1 / np.sqrt(n)))    # P(Zn ≥ t | Y=1)

# Define prevalence range
p = np.linspace(0, 1, 200)

# Compute PPV and NPV
PPV = (sens * p) / (sens * p + (1 - spec) * (1 - p))
NPV = (spec * (1 - p)) / (spec * (1 - p) + (1 - sens) * p)

# --- Plot styling (match previous purple & green figure) ---

# Colors and linestyles
color_ppv = 'purple'
color_npv = 'green'
linestyles = ['-', '--']

plt.figure(figsize=(8.5, 5))

# Plot PPV (purple solid)
plt.plot(p, PPV, color=color_ppv, linestyle=linestyles[0], linewidth=2.0,
         label='PPV', alpha=0.9)

# Plot NPV (green dashed)
plt.plot(p, NPV, color=color_npv, linestyle=linestyles[1], linewidth=2.0,
         label='NPV', alpha=0.9)

# Labels and styling
plt.xlabel('Prevalence $p$', fontsize=12)
plt.ylabel('Predictive Value', fontsize=12)
plt.title(f'PPV and NPV vs Prevalence ($t$ = 125, $n$ = 25)',
          fontsize=13, weight='bold')
plt.grid(True)
plt.legend(frameon=True, shadow=False, fontsize=9, loc='lower right', ncol=2)
plt.ylim(-0.05, 1.05)
plt.xlim(0, 1)
plt.tight_layout()
plt.show()