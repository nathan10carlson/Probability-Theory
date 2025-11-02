import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm

# Load MCMC results
data = np.load('thtas_results.npz')
print(data.files)  # shows what's inside

# Extract thtas array
thtas = data['thtas']
# extracting num samples (without burn in)
num_samples = thtas.shape[1]

# Determine which theta to use (you can use 0-8)
theta_to_test = 2
theta_samples = thtas[0,:]

# Dterming sample variance of the entire thing (S)
theta_variance = np.var(theta_samples, ddof=1)  # ddof=1 for sample variance
print(f'The variance of Theta {theta_to_test} is {theta_variance}')

def determine_batch_means (num_batches, theta_samples, num_samples):
    n = num_samples
    batch_size = n // num_batches  # integer division

    batch_means = np.zeros(num_batches)
    for i in range(num_batches):
        start = i * batch_size  # starting index of the current batch

        # For all batches except the last one
        if i < num_batches - 1:
            end = (i + 1) * batch_size  # usally this takes exactly batch_size samples
        else:
            end = n  # take everything for last batch, its okay if smaller

        # Compute mean of the current batch
        batch_means[i] = np.mean(theta_samples[start:end])
    return batch_means, batch_size

# how many batches to make
max_batch = 5000
tao = np.zeros(max_batch - 30)
for num_batches in range(30,max_batch):
    batch_means, batch_size = determine_batch_means (num_batches, theta_samples, num_samples)
    batch_mean_var = np.var(batch_means, ddof=1)
    tao[num_batches-30] = batch_size * batch_mean_var # i will divide everything by S^2 at the end

tao = tao / theta_variance

# creating x axis
x_batches = np.arange(30, max_batch)

plt.figure(figsize=(10, 5))
plt.plot(x_batches, tao, marker='.', linestyle='-', color='purple', label=r'$\tau$')
plt.xlabel('Number of batches', fontsize=12)
plt.ylabel(r'Normalized batch-mean variance ($\tau$)', fontsize=12)
plt.title(rf'$\tau$ for MCMC Variance ($\theta_{{{theta_to_test}}}$)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(x_batches[1000:], tao[1000:], marker='.', linestyle='-', color='purple', label=r'$\tau$')
plt.xlabel('Number of batches', fontsize=12)
plt.ylabel(r'Normalized batch-mean variance ($\tau$)', fontsize=12)
plt.title(rf'$\tau$ for MCMC Variance (num batches > $1,000$) ($\theta_{{{theta_to_test}}}$)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()