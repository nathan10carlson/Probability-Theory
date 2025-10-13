import numpy as np
import matplotlib.pyplot as plt

# Random Walks
# Specify how long the random walk is, the number of trials, the standard deviation,
# and the option to plot individual runs.
random_walk_length = 100
number_of_trials = 100000
std_dev = 1
num_bins = 1000
plot_option = False
compare_gaussian = True

def random_walk(step_size: float=1, num_walks: int=100, mean: float=0.0, std: float=1.0, show_plot=False):
    x = np.linspace(0, num_walks, num_walks+1)
    y = np.zeros_like(x)
    # begins at the origin
    current_state = 0
    for i in range(num_walks):
        rand = np.random.normal(mean, std)
        current_state = current_state + rand
        y[i+1] = current_state
    if show_plot is True:
        plt.plot(x, y)
        plt.show()
        plt.xlabel("Number of walks")
        plt.ylabel('Position')
    return y[num_walks-1]

random_walk(num_walks=random_walk_length, show_plot=plot_option, std=std_dev)

# This is the loop for running many tests and tracking what happens
tracking_results = []
counter = 0
while counter < number_of_trials:
    ending_val = random_walk(num_walks=random_walk_length, std=std_dev, show_plot=plot_option)
    tracking_results.append(ending_val)
    counter += 1
    print('% completed: {}'.format(counter/number_of_trials*100))

plt.hist(tracking_results, bins=int(num_bins), density=True, alpha=0.6, label="Random Walk Endpoints")

scaling_factor = .005
if compare_gaussian is True:
    mu = 0
    sigma_end = np.sqrt(random_walk_length) * std_dev   # variance grows with steps! really important
    x_vals = np.linspace(min(tracking_results), max(tracking_results), 1000)
    y = (1/(sigma_end * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_vals - mu)/sigma_end)**2)
    plt.plot(x_vals, y, 'r-', lw=2, label="Gaussian PDF")

plt.xlabel('Final Position')
plt.ylabel('Probability Density')
number_of_trials = 100000
plt.title(f"Random Walk with {random_walk_length} Steps ($10^{{{int(np.log10(number_of_trials))}}}$ trials, $\sigma =${std_dev})")
plt.legend()
plt.savefig('random_walk.png')
plt.show()
