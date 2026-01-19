import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.ndimage import gaussian_filter1d

# Loeading MCMC results
data = np.load('thtas_results.npz')
print(data.files)  # shows available arrays in the file

# Extract thtas array
thtas = data['thtas']
num_samples = thtas.shape[1]
print(f'Number of samples: {num_samples}')

# Choose which theta to analyze
theta_to_test = 2  # analyzes +1 theta
theta_samples = thtas[theta_to_test, :]

# Compute sample variance (S^2)
theta_variance = np.var(theta_samples, ddof=1)
print(f'Variance of Theta {theta_to_test}: {theta_variance:.6f}')


# Batch mean function that will be used in the loop
def determine_batch_means(num_batches, samples):
    n = len(samples)
    batch_size = n // num_batches

    # Trim remainder so batches are equal
    trimmed_n = batch_size * num_batches
    trimmed_samples = samples[:trimmed_n]

    batch_means = np.zeros(num_batches)
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_means[i] = np.mean(trimmed_samples[start:end])

    return batch_means, batch_size


# Compute normalized tau as a function of batch size
min_batch_size = 50  # should be at least a few autocorrelation times
max_batch_size = num_samples // 100  # wnaueinf batch size
batch_sizes = np.arange(min_batch_size, max_batch_size, step=50)

tau = np.zeros(len(batch_sizes))

for idx, batch_size in enumerate(batch_sizes):
    num_batches = num_samples // batch_size
    if num_batches < 2:  # too few batches
        tau[idx] = np.nan
        continue # Chat gpt added this part

    batch_means, _ = determine_batch_means(num_batches, theta_samples)
    batch_var = np.var(batch_means, ddof=1)
    tau[idx] = batch_size * (batch_var / theta_variance)  # normalized
#print(tau)
#print(np.min(tau))
#print(np.max(tau))
# Plot tau across batch sizes
plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, tau, marker='.', color='purple', label=r'$\tau$')
plt.xlabel(r'Batch size ($l$)', fontsize=12)
plt.ylabel(r'Normalized batch-mean variance ($\tau$)', fontsize=12)
plt.title(rf'$\tau$ for MCMC Variance ($\theta_{{{theta_to_test+1}}}$)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()



tau_smooth = gaussian_filter1d(tau, sigma=30)  # sigma controls smoothness

plt.figure(figsize=(10,5))
plt.plot(batch_sizes, tau, color='lightgray', alpha=0.5, label='Raw τ')
plt.plot(batch_sizes, tau_smooth, color='purple', label='Gaussian-smoothed τ')
plt.xlabel(r'Batch size ($l$)', fontsize=12)
plt.ylabel(r'Normalized batch-mean variance ($\tau$)', fontsize=12)
plt.title(rf'$\tau$ for MCMC Variance ($\theta_{{{theta_to_test+1}}}$)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Leftover zoomed in plots (not helpful rn)
plt.figure(figsize=(10, 5))
mask = batch_sizes > 1000
plt.plot(batch_sizes[mask], tau[mask], marker='.', color='purple', label=r'$\tau$ (zoomed)')
plt.xlabel('Batch size (samples per batch)', fontsize=12)
plt.ylabel(r'Normalized batch-mean variance ($\tau$)', fontsize=12)
plt.title(rf'$\tau$ for MCMC Variance (Batch size > 1000) ($\theta_{{{theta_to_test+1}}}$)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# === 95% Confidence Interval using batch means ===
num_batches_for_T = 950000 // 2500
batch_means, batch_size = determine_batch_means(num_batches_for_T, theta_samples)

X_bar = np.mean(batch_means)
SE = np.std(batch_means, ddof=1) / np.sqrt(num_batches_for_T)
df = num_batches_for_T - 1
t_value = t.ppf(1 - 0.05 / 2, df)

CI_lower = X_bar - t_value * SE
CI_upper = X_bar + t_value * SE

print(f"95% CI for θ_{theta_to_test+1}: [{CI_lower:.4f}, {CI_upper:.4f}]")

# === Plot θ samples with mean and 95% CI ===
x_axis = np.arange(num_samples)
plt.figure(figsize=(10, 5))
plt.plot(x_axis, theta_samples, label=r'$\theta$ samples', color='blue', alpha=0.2)
plt.axhline(X_bar, color='red', linestyle='--', label=f'Mean = {X_bar:.4f}')
plt.fill_between(x_axis, CI_lower, CI_upper, color='black', alpha=0.8, label='95% CI')

plt.xlabel('Sample index')
plt.ylabel(r'$\theta$ value')
plt.title(rf'$\theta_{{{theta_to_test+1}}}$ samples with 95% CI (batch means)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()