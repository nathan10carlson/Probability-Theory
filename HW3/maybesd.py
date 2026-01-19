import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.stattools import acf

# === Load MCMC results ===
data = np.load('thtas_results.npz')
thtas = data['thtas']
num_samples = thtas.shape[1]

# Choose which theta to analyze
theta_to_test = 2
theta_samples = thtas[theta_to_test, :]

# Compute overall sample variance
theta_variance = np.var(theta_samples, ddof=1)

# === Estimate autocorrelation time (optional but helpful) ===
acf_vals = acf(theta_samples, fft=True, nlags=1000)
plt.figure(figsize=(8,4))
plt.plot(acf_vals)
plt.title(f'Autocorrelation function for θ_{theta_to_test+1}')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid(True)
plt.show()

# Rough estimate of autocorrelation time
tau_int = 1 + 2 * np.sum(acf_vals[1:])  # integrated autocorrelation time
print(f"Estimated autocorrelation time: {tau_int:.2f} samples")

# === Batch mean function ===
def determine_batch_means(num_batches, samples):
    n = len(samples)
    batch_size = n // num_batches
    trimmed_n = batch_size * num_batches
    trimmed_samples = samples[:trimmed_n]
    batch_means = np.mean(trimmed_samples.reshape(num_batches, batch_size), axis=1)
    return batch_means, batch_size

# === Log-spaced batch sizes from a few autocorr times to ~5% of samples ===
min_batch_size = max(10, int(5 * tau_int))
max_batch_size = max(min_batch_size + 1, num_samples // 20)
batch_sizes = np.unique(np.round(np.logspace(np.log10(min_batch_size),
                                             np.log10(max_batch_size), 50)).astype(int))

tau = np.zeros(len(batch_sizes))

for idx, batch_size in enumerate(batch_sizes):
    num_batches = num_samples // batch_size
    if num_batches < 10:  # require at least 10 batches for stability
        tau[idx] = np.nan
        continue
    batch_means, _ = determine_batch_means(num_batches, theta_samples)
    batch_var = np.var(batch_means, ddof=1)
    tau[idx] = batch_size * (batch_var / theta_variance)

# === Smooth tau for visualization ===
tau_smooth = gaussian_filter1d(tau, sigma=3)

# === Plot τ ===
plt.figure(figsize=(10,5))
plt.plot(batch_sizes, tau, color='lightgray', alpha=0.6, label='Raw τ')
plt.plot(batch_sizes, tau_smooth, color='purple', label='Smoothed τ')
plt.xlabel('Batch size (samples per batch)')
plt.ylabel(r'Normalized batch-mean variance ($\tau$)')
plt.title(f'Normalized τ vs Batch Size for θ_{theta_to_test+1}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()