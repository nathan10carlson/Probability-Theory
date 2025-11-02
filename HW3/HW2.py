# %HW2, problem 4

# %note: the data in this problem was produced using 
# %parameters thta = [1; 1; 0.1; 0.1; 10; 10; 1; 1]. 
# %The z vector was obtained by running the solver with 
# %these parameters, and then adding Gaussian errors
# %of standard deviation 0.005 to the solution vector.

# %how close is your posterior mean to thta = [1; 1; 0.1; 0.1; 10; 10; 1; 1]?

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm

# this is copy-pasted from the code provided to you.
def solve_Poisson(thta):
    # % Inputs:
    # % theta = column vector of stiffnesses
    # % Outputs:
    # % x = fine grid points, ux = solution vector at fine grid points,
    # % z = coarse grid points, uz = solution vector at coarse grid points

    # %define node set (nodes are boundaries at which the stiffness changes)
    thta = np.asarray(thta).reshape(-1)  # ensure 1D
    N = thta.size
    xnodes = np.linspace(0, 1, N + 1)
    dx = np.diff(xnodes)

    # %define constant force
    f = 1  # constant source, or force
    nPerLayer = 100  # resolution

    # %global sums
    R0 = np.sum(dx / thta)
    R1 = np.sum((xnodes[1:] ** 2 - xnodes[:-1] ** 2) / (2 * thta))
    J0 = -(f * R1) / R0

    # %precompute partial sums
    C0 = np.concatenate(([0], np.cumsum(dx / thta)))
    C1 = np.concatenate(([0], np.cumsum((xnodes[1:] ** 2 - xnodes[:-1] ** 2) / (2 * thta))))

    # %pre-allocate arrays
    nTot = N * nPerLayer + 1
    x = np.zeros(nTot)
    u = np.zeros(nTot)

    # %fill arrays
    pos = 0
    for m in range(N):
        # %local grid in layer m
        xm = np.linspace(xnodes[m], xnodes[m + 1], nPerLayer + 1)
        if m > 0:
            xm = xm[1:]
        nm = xm.size

        x[pos : pos + nm] = xm
        u[pos : pos + nm] = -J0 * (C0[m] + (xm - xnodes[m]) / thta[m]) - f * (
            C1[m] + (xm**2 - xnodes[m] ** 2) / (2 * thta[m])
        )
        pos += nm

    # %enforce endpoints
    u[0] = 0
    u[-1] = 0

    # %get solution at coarse points
    # MATLAB: x(20:50:end) -> Python zero-based slice starting at index 19 with step 50
    x_16 = x[19::50]
    u_16 = u[19::50]

    return x, u, x_16, u_16

plt.close('all')
data = np.load('Poisson_data.npz')  # expects variables 'x' and 'z'
x = np.squeeze(data['x'])
z = np.squeeze(data['z'])

# %define standard deviations for prior and likelihood
sig_pr = 2.0    # %prior standard deviation (for lognormal prior)
sig = 0.005     # %likelihood standard deviation

# %define lognormal prior
def prior(thta: np.ndarray) -> float:
    # thta is a 1-D vector
    # MATLAB lognpdf(x, mu, sigma) -> scipy lognorm(s=sigma, scale=exp(mu))
    return np.prod(lognorm.pdf(thta, s=sig_pr, scale=np.exp(0.0)))

# %define likelihood
def u_exact(thta: np.ndarray) -> np.ndarray:
    _, _, _, u16 = solve_Poisson(thta)
    return u16
def likelihood(thta: np.ndarray) -> float:
    return np.prod(norm.pdf(z - u_exact(thta), loc=0.0, scale=sig))

# %define posterior
def posterior(thta: np.ndarray) -> float:
    return likelihood(thta) * prior(thta)

# %% perform MCMC

# %define number of steps to run
steps = int(1e5)

# %define proposal step size
del_step = 0.1

# %choose initial parameters
thta = np.ones(8)   # %initial condition for Monte Carlo (1-D vector)

# %define time series of parameters
thtas = np.zeros((8, steps))

# %run the MCMC
for step in range(steps):

    # %ticker
    if step % int(1e3) == 0 and step > 0:
        print("percent complete: ", step / steps)

    # %record current parameter vector
    thtas[:, step] = thta
    
    # %propose a symmetric step
    thta_prop = np.abs(thta + del_step * np.random.randn(8))

    # %define acceptance ratio
    p_ratio = posterior(thta_prop) / posterior(thta)

    # %accept with probability min(1,p_ratio)
    if np.random.rand() < min(1.0, p_ratio):
        thta = thta_prop
        
    #note: this loop can be made faster by 
    #using logs of the posterior and saving 
    #previous posterior computations, but I 
    #prioritized readability.

# %% plot the time series and determine burn-in

# %plot time series
plt.figure(figsize=(14, 6))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    idx = np.arange(0, steps, 100, dtype=int)
    plt.plot(idx + 1, thtas[i, idx], linewidth=1.2)
    plt.title(f'time series of theta_{i+1}')
    plt.xlabel('step')
    plt.ylabel(f'theta_{i+1}')
plt.tight_layout()

# %cut off burn-in time
burn_in = int(5e4)   # %cut out the "unconverged" part of the time series
thtas = thtas[:, burn_in:]
np.savez('thtas_results.npz', thtas=thtas)
# %% compare MCMC results to the measurements

# %assemble the posterior means
thta_mean = np.mean(thtas, axis=1)

# %find the model solution at the mean parameters
x_fine, u_fine, _, _ = solve_Poisson(thta_mean)

# %compare to the solution at the "true" parameter vector
thta_true = np.array([1, 1, 0.1, 0.1, 10, 10, 1, 1])
x_true, u_true, _, _ = solve_Poisson(thta_true)
# %note: I did not give you access to parameters! 
# %how close are your results to these true values?

# %plot the means against the measurements
plt.figure()
plt.plot(x_fine, u_fine, '-')
plt.plot(x_true, u_true, '-.')
plt.plot(x, z, 'x')
plt.legend(['u at posterior mean', 'exact u', 'measurements'])
plt.xlabel('x')
plt.ylabel('u(x)')

# %% plot some joint densities

# %plot joint density of thta1 and thta2
plt.figure()
plt.hist2d(thtas[0, :], thtas[1, :], bins=20, density=True)
plt.title('joint density of theta1 and theta2')
plt.xlabel('theta1')
plt.ylabel('theta2')
plt.colorbar()

# %plot joint density of thta3 and thta4
plt.figure()
plt.hist2d(thtas[2, :], thtas[3, :], bins=20, density=True)
plt.title('joint density of theta3 and theta4')
plt.xlabel('theta3')
plt.ylabel('theta4')
plt.colorbar()

# %plot joint density of thta3 and thta4
plt.figure()
plt.hist2d(thtas[4, :], thtas[5, :], bins=20, density=True)
plt.title('joint density of theta5 and theta6')
plt.xlabel('theta5')
plt.ylabel('theta6')
plt.colorbar()

# %plot joint density of thta2 and thta6
plt.figure()
plt.hist2d(thtas[1, :], thtas[5, :], bins=20, density=True)
plt.title('joint density of theta2 and theta6')
plt.xlabel('theta2')
plt.ylabel('theta6')
plt.colorbar()

plt.show()