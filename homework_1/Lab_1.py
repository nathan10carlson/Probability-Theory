import numpy as np
import matplotlib.pyplot as plt


num_steps = 10**4
show_individual_plots = False
number_of_trials = 1
all_together_individual = True
compare_gaussian = True

# Initializing empty matrix for individual runs as rows
if all_together_individual is True:
    positions_for_runs = np.zeros((int(number_of_trials),int(num_steps+1)) )

class random_walks():
    def __init__(self, number_of_steps: int = 1000, plot_walk: bool=True):
        self.number_of_steps = number_of_steps
        self.plot_walk = plot_walk
        self.positions = None

    def rand_walk(self):
        start_pos = 0
        positions = np.zeros(self.number_of_steps+1)

        for i in range(self.number_of_steps):
            step = np.random.choice([-1, 1])
            positions[i+1] = positions[i] + step

        self.positions = positions # storing for later use

        if self.plot_walk is True:
            x = np.linspace(start_pos, start_pos+self.number_of_steps, self.number_of_steps+1 )
            plt.plot(x, positions)
            plt.xlabel('Step Number')
            plt.ylabel('Position')
            plt.title(f'Random Walk with {num_steps} Steps')
            plt.show()
        return positions

    def B(self, t: float = None, plot: bool = False):
        # Compute B(t) for a single point
        if t is not None:
            index = int(np.floor(t * self.number_of_steps))
            return self.positions[index] / np.sqrt(self.number_of_steps)

        # Plot the entire path if you tell it to
        if plot:
            times = np.linspace(0, 1, self.number_of_steps + 1)
            scaled_positions = self.positions / np.sqrt(self.number_of_steps)

            plt.plot(times, scaled_positions, label="Single Brownian Motion Path", color='g')
            plt.xlabel("t")
            plt.ylabel("B(t)")
            plt.title(f"Brownian Motion Approximation with {self.number_of_steps} steps")
            plt.legend()
            plt.show()


for i in range(number_of_trials):
    test_run = random_walks(number_of_steps=num_steps, plot_walk=show_individual_plots)
    positions = test_run.rand_walk()
    if all_together_individual is True:
        positions_for_runs[i,:] = positions

    # Plotting them all together

    x = np.linspace(0, 0 + num_steps, num_steps + 1)
    plt.plot(x, positions_for_runs[i,:])

plt.xlabel('Step Number')
plt.ylabel('Position')
plt.title(f'Random Walk with $10^{{{int(np.log10(num_steps))}}}$ Steps')
plt.show()

# Plotting histogram
plt.xlabel('Final Position')
plt.ylabel('Probability')
plt.hist(positions_for_runs[:, -1], bins=40, density=True, alpha=0.7, edgecolor="black")
if compare_gaussian is True:
    mu = 0
    sigma_end = np.sqrt(num_steps) # variance grows with steps! really important
    x_vals = np.linspace(min(positions_for_runs[:,-1]), max(positions_for_runs[:,-1]), num_steps + 1)
    y = (1/(sigma_end * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_vals - mu)/sigma_end)**2)
    plt.plot(x_vals, y, 'r-', lw=2, label="Gaussian PDF")
    plt.title(
        f"Random Walk with $10^{int(np.log10(num_steps))}$ steps "
        f"($10^{int(np.log10(number_of_trials))}$ trials)"
    )
plt.show()

rw = random_walks(number_of_steps=10**4, plot_walk=True)
rw.rand_walk()

# Get single point
print(rw.B(0.25))  # B(t) at t=0.25

# Plot the whole path
rw.B(plot=True)
#test_run.combined_graph(positions)

