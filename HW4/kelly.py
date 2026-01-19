import numpy as np
import matplotlib.pyplot as plt
import random

# Values
starting_wealth = 100
b = 1.0       # winnings (1 dollar wager earns b dollars)
p = 0.75      # win probability
num_rounds = 50
num_sims = 10000

# Kelly Rule
def f_star(p, b):
    return p - (1 - p) / b

# Wealth update
def wager_once(current_wealth, f, p, b):
    win = random.random() < p      # True if win
    X = b if win else -1           # payoff relative to wager
    return current_wealth * (1 + f * X)

# List of Kelly fractions to test
kelly_factors = [0.25, 0.5, 1, 1.5, 2]

for k in kelly_factors:
    f = k * f_star(p, b)
    final_wealths = np.zeros(num_sims)

    # Run simulations
    for s in range(num_sims):
        wealth = starting_wealth
        for _ in range(num_rounds):
            wealth = wager_once(wealth, f, p, b)
        final_wealths[s] = wealth

    # Compute stats
    median_wealth = np.median(final_wealths)
    mean_wealth = np.mean(final_wealths)
    max_wealth = np.max(final_wealths)
    min_wealth = np.min(final_wealths)
    num_losses = np.sum(final_wealths < starting_wealth)
    percent_losses = num_losses / num_sims * 100
    num_doubled = np.sum(final_wealths >= 2 * starting_wealth)
    percent_doubled = num_doubled / num_sims * 100
    num_tripled = np.sum(final_wealths >= 3 * starting_wealth)
    percent_tripled = num_tripled / num_sims * 100

    # Print diagnostics
    print(f"Kelly factor: {k}")
    print(f"Median: ${median_wealth:,.2f}, Mean: ${mean_wealth:,.2f}")
    print(f"Min: ${min_wealth:,.2f}, Max: ${max_wealth:,.2f}")
    print(f"Losing simulations: {num_losses} ({percent_losses:.2f}%)")
    print(f"Doubled money: {num_doubled} ({percent_doubled:.2f}%)")
    print(f"Tripled money: {num_tripled} ({percent_tripled:.2f}%)")
    print("-" * 60)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(final_wealths, bins=75, density=True, alpha=0.7, edgecolor='black')
    plt.axvline(median_wealth, color='red', linestyle='--', linewidth=2,
                label=f'Median: ${median_wealth:,.0f}')
    plt.xlabel(rf"Final Wealth After {num_rounds} Rounds")
    plt.ylabel("Density")
    plt.title(rf"Distribution of Final Wealth ({num_sims} Sims, Kelly Factor = {k})")
    plt.grid(True)
    plt.xscale('log')
    plt.legend()
    plt.show()