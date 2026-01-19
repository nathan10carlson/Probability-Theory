import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# --- Load MCMC results ---
data = np.load('thtas_results.npz')
thtas = data['thtas']
num_samples = thtas.shape[1]

# Choose theta to analyze
theta_to_test = 5
theta_samples = thtas[theta_to_test, :]

# Sample variance of the full chain
S2 = np.var(theta_samples, ddof=1)

# --- Function to compute batch means ---
def determine_batch_means(num_batches, samples):
    n = len(samples)
    batch_size = n // num_batches
    batch_means = np.zeros(num_batches)
    for i in range(num_batches):
        start = i * batch_size
        end = (i+1) * batch_size if i < num_batches-1 else n
        batch_means[i] = np.mean(samples[start:end])
    return batch_means, batch_size

# --- Estimate tau for various batch sizes ---
min_batches = 30
max_batches = 2000
taus = []

for num_batches in range(min_batches, max_batches):
    batch_means, batch_size = determine_batch_means(num_batches, theta_samples)
    S_batch2 = np.var(batch_means, ddof=1)
    tau = batch_size * S_batch2 / S2
    taus.append(tau)

taus = np.array(taus)

# --- Pick batch size where tau is roughly stable (~1) ---
# We'll take the batch size where tau first falls below 1.1 (or close)
stable_idx = np.where(taus <= 1.1)[0]
if len(stable_idx) > 0:
    optimal_num_batches = min_batches + stable_idx[0]
else:
    optimal_num_batches = min_batches  # fallback

print(f"Optimal number of batches: {optimal_num_batches}")
batch_means, batch_size = determine_batch_means(optimal_num_batches, theta_samples)

# --- Compute Student t 95% CI ---
X_bar = np.mean(batch_means)
SE = np.std(batch_means, ddof=1) / np.sqrt(optimal_num_batches)
df = optimal_num_batches - 1
t_val = t.ppf(1 - 0.05/2, df)
CI_lower = X_bar - t_val * SE
CI_upper = X_bar + t_val * SE
print(f"95% CI for theta: [{CI_lower:.4f}, {CI_upper:.4f}]")

# --- Plot with CI ---
x_axis = np.arange(num_samples)
plt.figure(figsize=(10,5))
plt.plot(x_axis, theta_samples, label=r'$\theta$ samples', alpha=0.6)
plt.axhline(X_bar, color='red', linestyle='--', label=f'Mean = {X_bar:.4f}')
plt.fill_between(x_axis, CI_lower, CI_upper, color='orange', alpha=0.3, label='95% CI')
plt.xlabel('Sample index')
plt.ylabel(r'$\theta$ value')
plt.title('Theta samples with 95% CI from batch means')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Optional: plot tau vs batch size ---
plt.figure(figsize=(10,5))
plt.plot(np.arange(min_batches, max_batches), taus, color='purple')
plt.axhline(1, color='black', linestyle='--', label=r'$\tau=1$')
plt.xlabel('Number of batches')
plt.ylabel(r'$\tau$')
plt.title(r'$\tau$ vs Number of batches')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()