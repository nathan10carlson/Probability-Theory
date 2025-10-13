import numpy as np
import matplotlib.pyplot as plt

# Analytic expected value function
def expected_wager(p):
    return 1 / (2*p - 1)

# Generate p values from 0.51 to 1
p_values = np.linspace(0.505, 1, 500)
E_values = expected_wager(p_values)

# Plot
plt.figure(figsize=(8,5))
plt.plot(p_values, E_values, label=r'$\mathbb{E}[X] = \frac{1}{2p-1}$', color='blue')
plt.axvline(0.5, color='red', linestyle='--', label='Divergence at p=0.5')
plt.xlabel('Probability of Winning (p)')
plt.ylabel(r'Expected Cumulative Wager E[X] $(\$)$ ')
plt.title('Expected Wager Before Winning in Doubling or Nothing Game')
plt.legend()
plt.grid(True)
plt.ylim(0, np.max(E_values)*1.1)  # optional: scale y-axis nicely
plt.show()

def doubling_bet_simulation(p, wallet, num_trials=1000):
    successes = 0
    for _ in range(num_trials):
        wager = 1
        cumulative = 0
        while cumulative + wager <= wallet:
            cumulative += wager
            if np.random.rand() < p:  # win
                successes += 1
                break
            wager *= 2
    success_rate = successes / num_trials
    return success_rate

# Parameters
wallet = 5
num_trials = 5000
p_values = np.arange(0.05, 1.01, 0.05)  # p from 0.05 to 1 in steps of 0.05
success_rates = []

# Run simulation for each p
for p in p_values:
    rate = doubling_bet_simulation(p, wallet, num_trials)
    success_rates.append(rate)
# Plot histogram (bar plot)
plt.figure(figsize=(10,5))
plt.bar(p_values, success_rates, width=0.045, color='skyblue', edgecolor='black')
plt.xlabel("Probability of Winning (p)")
plt.ylabel("Success Rate")
plt.title(f"Success Rate vs Winning Probability (Wallet=${wallet})")
plt.ylim(0,1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()