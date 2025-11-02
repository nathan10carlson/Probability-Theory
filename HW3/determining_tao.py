import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm, t

# Load MCMC results
data = np.load('thtas_results.npz')
print(data.files)  # shows what's inside

# Extract thtas array
thtas = data['thtas']
# extracting num samples (without burn in)
num_samples = thtas.shape[1]
print(f'number of samples: {num_samples}')

# Determine which theta to use (you can use 0-7)
theta_to_test = 2 # remember, you will really test theta +1 for whatever you put
theta_samples = thtas[theta_to_test,:]

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

####################
run_tau = True

if run_tau == True:
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
    plt.title(rf'$\tau$ for MCMC Variance ($\theta_{{{theta_to_test+1}}}$)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x_batches[1000:], tao[1000:], marker='.', linestyle='-', color='purple', label=r'$\tau$')
    plt.xlabel('Number of batches', fontsize=12)
    plt.ylabel(r'Normalized batch-mean variance ($\tau$)', fontsize=12)
    plt.title(rf'$\tau$ for MCMC Variance (num batches > $1,000$) ($\theta_{{{theta_to_test+1}}}$)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

## SDeterming student T distribution
number_of_batches_for_T = 4500
batch_means, batch_size = determine_batch_means(number_of_batches_for_T, theta_samples, num_samples)
SE = np.std(batch_means, ddof=1) / np.sqrt(number_of_batches_for_T)

df = number_of_batches_for_T - 1
t_value = t.ppf(1 - 0.05/2, df)  # two-sided 95% CI CHAT GPT caught this
X_bar = np.mean(batch_means)

CI_lower = X_bar - t_value * SE
CI_upper = X_bar + t_value * SE

print(f"95% CI for theta: [{CI_lower:.4f}, {CI_upper:.4f}]")

# x-axis: sample index
x_axis = np.arange(num_samples)

# Plot the theta samples
plt.figure(figsize=(10,5))
plt.plot(x_axis, theta_samples, label=r'$\theta$ samples', color='blue', alpha=0.6)

# Add horizontal line for mean
plt.axhline(X_bar, color='red', linestyle='--', label=f'Mean = {X_bar:.4f}')

# Add shaded area for 95% CI
plt.fill_between(x_axis, CI_lower, CI_upper, color='orange', alpha=0.3, label='95% CI')

plt.xlabel('Sample index')
plt.ylabel(r'$\theta$ value')
plt.title(r'Theta samples with 95% CI from batch means')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()