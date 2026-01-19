import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

num_steps = int(1e5)
num_trials = int(1e5)

# Convert num_steps for plot labels
steps_label = rf"$10^{{{int(np.log10(num_steps))}}}$"

# ---------------------------------------
def single_walk(num_steps):
    steps = np.random.choice([-1, 1], size=num_steps)
    scaling = np.linspace(1, num_steps+1, num_steps)   # keep your style
    walk = np.cumsum(steps) / np.sqrt(scaling)
    return walk

walk_result = single_walk(num_steps)

# ---------------------------------------
def distirbution(num_steps, num_trials):
    steps_matrix = np.random.choice([-1, 1], size=(num_trials, num_steps))
    final_state = np.sum(steps_matrix, axis=1) / np.sqrt(num_steps)

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(final_state, bins=30, density=True, alpha=0.6, edgecolor='black')

    # normal curve
    x = np.linspace(-4, 4, 400)
    plt.plot(x, norm.pdf(x, 0, 1), 'purple', lw=2)

    # title + labels
    plt.title(f"Distribution of Final State\n{num_trials:,} trials, {steps_label} steps each")
    plt.xlabel(f"Final Position")
    plt.ylabel("Density")
    plt.grid(alpha=0.2)
    plt.show()

    return final_state

# Random walk plot
plt.figure(figsize=(8, 5))
plt.plot(walk_result, lw=1)
plt.title(f"Random Walk with Scaling ({steps_label} steps)")
plt.xlabel("Step")
plt.ylabel(f"Normalized Location")
plt.grid(alpha=0.2)
plt.show()

# Run trials
running_trials = distirbution(num_steps, num_trials)


# Plot random walks on the same figure
plt.figure(figsize=(8, 5))

for i in range(4):
    walk_i = single_walk(num_steps)
    plt.plot(walk_i, lw=1, label=f"Walk {i+1}", )

plt.title(f"Comparing Random Walks ({steps_label} steps)")
plt.xlabel("Step(n)")
plt.ylabel(f"Normalized Location")
plt.grid(alpha=0.2)
plt.legend()
plt.show()