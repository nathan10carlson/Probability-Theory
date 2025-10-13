# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%% SCHLOGL MODEL SIMULATOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %initialize simulation

import numpy as np
import matplotlib.pyplot as plt

# %define physical constants
p = 1
q = 2
k1 = 3
k2 = 0.6
k3 = 0.25
k4 = 2.95
V = 25

# %define reaction rates
def lam1(i):  # %rate A+2S->3S
    return p * k1 * i * (i - 1) / V

def lam2(i):  # %rate 3S->A+2S
    return k2 * i * (i - 1) * (i - 2) / (V ** 2)

def lam3(i):  # %rate B->S
    return q * k3 * V

def lam4(i):  # %rate S->B
    return k4 * i

# %define simulation parameters
tau = 0.01   # %time step for simulations
N = 10**6    # %number of time steps to simulate

# %initialize data vectors
population = np.zeros(N, dtype=int)   # %vector of S species populations by time step

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %begin simulation of the Schlogl model (using tau-leaping)

i = 0   # %initial population
for n in range(1, N):

    # %display simulation progress
    if n % int(1e4) == 0:
        print(f"percent complete = ...{n / N}")

    # %simulate next time step
    i_inc = np.random.poisson(tau * (lam1(i) + lam3(i)))
    i_dec = np.random.poisson(tau * (lam2(i) + lam4(i)))
    i = max(i + i_inc - i_dec, 0)

    # %record current population
    population[n] = i

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %plot the Schlogl model simulation results

plt.plot(np.arange(N) * tau, population, '-b')
plt.xlabel('time t')
plt.ylabel('population of species S')
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %construct the rate matrix, Q, for the Schlogl model

M = 180   # %choose cutoff, or maximum value of population

Q = np.zeros((M + 1, M + 1))   # %initialize rate matrix
for i_val in range(0, M):
    Q[i_val, i_val + 1] = lam1(i_val) + lam3(i_val)   # %total rate to increase i
for i_val in range(1, M + 1):
    Q[i_val, i_val - 1] = lam2(i_val) + lam4(i_val)   # %total rate to decrease i

Q = Q - np.diag(Q.sum(axis=1))   # %diagonal adjustment
