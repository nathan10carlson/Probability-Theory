import numpy as np
import matplotlib.pyplot as plt

class GamblersRuin():
    def __init__(self, prob: float = 0.5, number_of_games=30):
        self.prob = prob
        self.number_of_games = number_of_games

    def calc_expected_value(self):
        expected_value = 0
        wager_n = 0
        for n in range(1, self.number_of_games + 1):
            prob_n = (1 - self.prob)**(n - 1) * self.prob   # lose n-1 then win
            wager_n += 2**(n-1)                              # total wager if win on nth
            expected_value += prob_n * wager_n
        expected_value *= self.prob
        return expected_value

# Probabilities from 0.01 to 0.99 (im skipping 0 and 1)
p_vals = np.linspace(0.6, 0.99, 50)
avg_wager = np.zeros_like(p_vals)

for i, p in enumerate(p_vals):
    game = GamblersRuin(prob=p, number_of_games=750)
    avg_wager[i] = game.calc_expected_value()

plt.figure(figsize=(8,5))
plt.plot(p_vals, avg_wager, marker='o')
plt.title("Expected Wager for Double-or-Nothing Strategy")
plt.xlabel("Probability of Winning (p)")
plt.ylabel("Expected Total Wager (capped at 30 rounds)")
#plt.yscale('log')
plt.grid(True)
plt.show()