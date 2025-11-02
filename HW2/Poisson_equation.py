import numpy as np
from matplotlib import pyplot as plt



# Choosing a prior distribution
n_theta = 8
n_theta_min = .1
n_theta_max = 10
prior_distribution = np.random.uniform(low=n_theta_min, high=n_theta_max, size=n_theta)
print(prior_distribution)


