import matplotlib.pyplot as plt
import numpy as np


def birthday_paradox(num_people):
    prev_prob = 1
    collision_probs = np.zeros(num_people)
    for i in range(num_people):
        prev_prob *= (365 - i) / 365
        collision_probs[i] = 1 - prev_prob  # store collision probability

    x = np.arange(1, num_people + 1)
    return x, collision_probs

######################################
x_vals, y_vals = birthday_paradox(50)#
######################################

plt.plot(x_vals, y_vals * 100, linestyle='-', color='navy')
plt.xlabel("Number of people")
plt.ylabel("Probability (%)")
plt.title(r'Birthday Paradox: Collision Probability $\geq$1')
plt.yticks(range(0, 101, 10))
plt.show()