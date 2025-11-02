import numpy as np
import matplotlib.pyplot as plt

# Target unnormalized density: pi(x) ∝ exp(-x^4 + x^2)
def log_target_density(x):
    return -x**4 + x**2  # We use log-density for numerical stability

# Metropolis-Hastings parameters
num_samples = 50000       # total number of samples
burn_in = 1000            # how many to discard as burn-in
sigma = 1.0               # proposal std deviation

# Initialize
samples = np.zeros(num_samples)
x_current = 0.0           # start at x=0
log_p_current = log_target_density(x_current)


# Main Metropolis-Hastings loop
for t in range(1, num_samples):
    # Propose new candidate
    x_proposal = np.random.normal(loc=x_current, scale=sigma)
    log_p_proposal = log_target_density(x_proposal)

    # Acceptance probability (log scale)
    log_alpha = log_p_proposal - log_p_current
    if np.log(np.random.rand()) < log_alpha:
        x_current = x_proposal
        log_p_current = log_p_proposal

    samples[t] = x_current

# Remove burn-in samples
samples_post = samples[burn_in:]

# Estimate expectation as an example
mean_estimate = np.mean(samples_post)
print(f"Estimated E[x] ≈ {mean_estimate:.4f}")

# Plot the trace (chain over time)
plt.figure(figsize=(12, 4))
plt.plot(samples)
plt.axvline(burn_in, color='red', linestyle='--', label='Burn-in cutoff')
plt.title("Metropolis–Hastings Trace Plot")
plt.xlabel("Iteration")
plt.ylabel("x")
plt.legend()
plt.show()

# Plot the histogram vs true (unnormalized) target shape
x_grid = np.linspace(-3, 3, 500)
unnormalized_target = np.exp(log_target_density(x_grid))

plt.figure(figsize=(8, 5))
plt.hist(samples_post, bins=50, density=True, alpha=0.6, label='MCMC samples')
plt.plot(x_grid, unnormalized_target / np.trapezoid(unnormalized_target, x_grid),
         'r-', label='True target (normalized)')
plt.title("Sampled Distribution vs Target")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()

