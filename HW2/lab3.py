import numpy as np
import matplotlib.pyplot as plt
import random

num_flips = 1000
prob_tail = .01 # true probability

print(f'The actual probability of getting tails is {prob_tail* 100:.2f}%')

num_tails = sum(random.random() < prob_tail for _ in range(num_flips))

print(f'There were {num_tails} tails.')
print(f'There were {num_flips - num_tails} heads.')
print(f'The ratio is {num_tails / num_flips * 100:.2f}% were tails.')

# Posterior over theta
theta_vals = np.linspace(0, 1, 1000)
posterior_unnormalized = theta_vals**num_tails * (1 - theta_vals)**(num_flips - num_tails)

# Normalize to make it a PDF (Thanks ChatGPT for helping with integration to check #
area = np.trapezoid(posterior_unnormalized, theta_vals)   # numerical integral over [0,1]
posterior_pdf = posterior_unnormalized / area

# Plot
plt.plot(theta_vals, posterior_pdf, linestyle='--', color='purple')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta | D) p(\theta)$')
#plt.yscale('log')
plt.title(f'Posterior PDF of θ ({num_flips} flips, true p={prob_tail:.2f})')
plt.grid(True)
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.show()

# Check: integral should be 1 (approximately)
print("Check normalization:", np.trapezoid(posterior_pdf, theta_vals))