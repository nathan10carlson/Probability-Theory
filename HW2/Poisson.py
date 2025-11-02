from traceback import print_tb

import numpy as np
import matplotlib.pyplot as plt
import time as time

# ==================== Load data ====================

def load_data(filename='Poisson_data.npz'):
    """Load measurement data and append boundary points."""
    data = np.load(filename)
    x = data['x'].flatten()
    z = data['z'].flatten()
    data.close()

    # Append boundary points
    x_full = np.concatenate(([0], x, [1]))
    z_full = np.concatenate(([0], z, [0]))

    return x_full, z_full, x, z

# ==================== Poisson solver ====================

def solve_Poisson(theta, nPerLayer=100, nCoarse=16):
    """Solve 1D Poisson equation with piecewise-constant stiffness theta."""
    theta = np.asarray(theta).reshape(-1)
    N = theta.size
    xnodes = np.linspace(0, 1, N + 1)
    dx = np.diff(xnodes)
    f = 1  # constant source

    # Global flux quantities
    R0 = np.sum(dx / theta)
    R1 = np.sum((xnodes[1:]**2 - xnodes[:-1]**2) / (2 * theta))
    J0 = -(f * R1) / R0

    # Cumulative sums for each region
    C0 = np.concatenate(([0], np.cumsum(dx / theta)))
    C1 = np.concatenate(([0], np.cumsum((xnodes[1:]**2 - xnodes[:-1]**2) / (2 * theta))))

    # Fine grid solution
    nTot = N * nPerLayer + 1
    x = np.zeros(nTot)
    u = np.zeros(nTot)
    pos = 0

    for m in range(N):
        xm = np.linspace(xnodes[m], xnodes[m+1], nPerLayer + 1)
        if m > 0:
            xm = xm[1:]
        nm = xm.size
        x[pos:pos+nm] = xm
        u[pos:pos+nm] = -J0 * (C0[m] + (xm - xnodes[m]) / theta[m]) - f * (
            C1[m] + (xm**2 - xnodes[m]**2) / (2 * theta[m])
        )
        pos += nm

    # BCs
    u[0] = 0
    u[-1] = 0

    # Coarse sampling for measurement comparison
    step = len(x) // nCoarse
    x_coarse = x[step-1::step][:nCoarse]
    u_coarse = u[step-1::step][:nCoarse]

    return x, u, x_coarse, u_coarse

# ==================== Plot solution vs data ====================

def plot_Poisson(x, u, x_coarse, u_coarse, z):
    plt.figure(figsize=(8,5))
    plt.plot(x, u, label='Solver (fine grid)', color='blue')
    plt.plot(x_coarse, u_coarse, 'o', label='Solver (coarse points)', color='green')
    plt.plot(x_coarse, z, 'x', label='Observed data', color='red')
    plt.xlabel('x')
    plt.ylabel('Membrane deformation')
    plt.title('Poisson equation solution vs observed data')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_theta_2d(theta_post, idx1, idx2, bins=30, hexbin=False):
    """
    Plot a 2D histogram / joint posterior distribution of two theta parameters.

    Parameters:
    -----------
    theta_post : ndarray
        Posterior MCMC samples of shape (num_samples, n_theta)
    idx1, idx2 : int
        Indices of the two theta parameters to plot (0-based)
    bins : int
        Number of bins for histogram
    hexbin : bool
        If True, use hexbin plot; otherwise use plt.hist2d
    """
    x = theta_post[:, idx1]
    y = theta_post[:, idx2]

    plt.figure(figsize=(7, 5))

    if hexbin:
        plt.hexbin(x, y, gridsize=bins, cmap='Blues', mincnt=1)
        plt.colorbar(label='Counts')
    else:
        plt.hist2d(x, y, bins=bins, cmap='Blues', density=True)
        plt.colorbar(label='Density')

    plt.xlabel(f"$\\theta_{{{idx1 + 1}}}$")
    plt.ylabel(f"$\\theta_{{{idx2 + 1}}}$")
    plt.title(f"2D distribution of $\\theta_{{{idx1 + 1}}}$ vs $\\theta_{{{idx2 + 1}}}$")
    plt.grid(True)
    plt.show()
# Likelihood calculation

def log_likelihood(u_coarse):
    sigma = 0.005  # fixed noise std
    u_error = z - u_coarse
    # taking log
    return (-1/ (2 *sigma**2)) * np.sum((u_error)**2)

#####################################################
# Main Code
start_time = time.time()


# Load data
x_full, z_full, x_obs, z = load_data()

# Initial theta
n_theta = 8
theta = np.random.uniform(low=0.1, high=10.0, size=n_theta)
theta = np.array([1.0284,1.027,0.107,0.132,5.82,6.3,1.2,0.65])

# MCMC settings
num_steps = int(1e6)
#num_steps = 100
proposal_scale = .02 ###### DETERMINE VARIANCE of added noise HERE
prop_scale = .75 ### Determine how often things are accepted
plot_theta = True
plot_means = True
plot_joints = True

# Storage
theta_chain = np.zeros((num_steps, n_theta))
log_like_chain = np.zeros(num_steps)
accept_count = 0

# Initial solve
_, _, _, u_coarse = solve_Poisson(theta)
prev_log_like = log_likelihood(u_coarse)

# Running loop
for i in range(num_steps):
    update_theta = np.random.normal(loc=0, scale=proposal_scale, size=n_theta)
    proposed_theta = np.abs(theta + update_theta)  # make sure its postive

    _, _, _, u_coarse_prop = solve_Poisson(proposed_theta)
    proposed_log_like = log_likelihood(u_coarse_prop)

    log_pi_j = proposed_log_like - prev_log_like

    if np.log(prop_scale *np.random.rand()) < log_pi_j:
        theta = proposed_theta
        prev_log_like = proposed_log_like
        accept_count += 1

    theta_chain[i] = theta
    log_like_chain[i] = prev_log_like

# Results

# tracking acceptance rate
accept_rate = accept_count / num_steps
print(f"Acceptance rate: {accept_rate:.2%}")

#removing the burnin time
burn_in = int(0.25 * num_steps)
theta_post = theta_chain[burn_in:, :]

# Posterior means
posterior_mean = np.mean(theta_post, axis=0)
print("Posterior mean of θ:", posterior_mean)
posterior_median = np.median(theta_post, axis=0)

# Plots for theta vals
if plot_theta == True:
    for j in range(int(n_theta)):
        plt.figure(figsize=(10, 3))
        plt.plot(theta_chain[:, j], lw=0.5)
        plt.ylabel(f"$\\theta_{{{j+1}}}$")
        plt.xlabel("MCMC iteration")
        plt.title(f"Value of $\\theta_{{{j+1}}}$ over time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

#Histograms

if plot_means == True:
    # First half
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4.5))
    axes1 = axes1.ravel()

    for j in range(4):
        axes1[j].hist(theta_post[:, j], bins=30, density=True, color='steelblue', alpha=0.7)
        axes1[j].axvline(posterior_mean[j], color='red', linestyle='--', label='Posterior mean')
        axes1[j].set_title(f"$\\theta_{{{j+1}}}$")
        axes1[j].grid(True)
        axes1[j].legend()

    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
    axes2 = axes2.ravel()
    #second half
    for j in range(4, n_theta):
        axes2[j-4].hist(theta_post[:, j], bins=30, density=True, color='steelblue', alpha=0.7)
        axes2[j-4].axvline(posterior_mean[j], color='red', linestyle='--', label='Posterior mean')
        axes2[j-4].set_title(f"$\\theta_{{{j+1}}}$")
        axes2[j-4].grid(True)
        axes2[j-4].legend()
    plt.tight_layout()
    plt.show()

# Plotting all together
end_time = time.time()
print(f"Runtime: {end_time - start_time}")
x_final, u_final, x_coarse_final, u_coarse_final = solve_Poisson(posterior_mean)
plot_Poisson(x_final, u_final, x_coarse_final, u_coarse_final, z)

x_final, u_final, x_coarse_final, u_coarse_final = solve_Poisson(posterior_median)
plot_Poisson(x_final, u_final, x_coarse_final, u_coarse, z)

# Plot all unique theta pairs
if plot_joints == True:
    pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 5), (4, 6), (4, 7),
        (5, 6), (5, 7),
        (6, 7)
    ]

    for idx1, idx2 in pairs:
        plot_theta_2d(theta_post, idx1, idx2, bins=30, hexbin=False)

    x_full, z_full, x_obs, z = load_data()
    print(x_full)
    print(z_full)
    print(x_obs)
    print(z)