import numpy as np
import matplotlib.pyplot as plt

num_steps = int(1e4)
red = 2
green = 1
starting_balls = red + green

def polyas_urn_trial(num_steps, red, green, starting_balls):
    red_local = red
    green_local = green

    red_steps = np.zeros(num_steps+1)
    red_steps[0] = red_local  # initial number of red balls
    random_decisions = np.random.rand(num_steps)

    for steps in range(num_steps):
        prob_red_added = red_local / (red_local + green_local)
        random_val = random_decisions[steps]

        if prob_red_added > random_val:
            red_local += 1
        else:
            green_local += 1
        red_steps[steps+1] = red_local

    return red_steps

def polyas_run_no_storage(num_steps, red, green, starting_balls):
    red_local = red
    green_local = green
    random_decisions = np.random.rand(num_steps)

    for steps in range(num_steps):
        prob_red_added = red_local / (red_local + green_local)
        random_val = random_decisions[steps]

        if prob_red_added > random_val:
            red_local += 1
        else:
            green_local += 1

    return red_local / (red_local + green_local)

# ---------------- Trajectory plot ----------------
plt.figure(figsize=(8,5))
for trial in range(5):
    red_steps = polyas_urn_trial(num_steps, red, green, starting_balls)
    total_balls = np.arange(0, num_steps+1) + starting_balls
    ratio_of_red = red_steps / total_balls
    plt.plot(ratio_of_red, label=f'Trial {trial+1}')

# Mark the initial fraction
initial_fraction = red / starting_balls
plt.axhline(initial_fraction, color='black', linestyle='--', label='Initial Fraction')
plt.text(num_steps*0.95, initial_fraction + 0.02, f'Start={initial_fraction:.2f}',
         horizontalalignment='right', color='black')

plt.xlabel("Step")
plt.ylabel("Fraction of Red Balls")
plt.title(rf"Polya's Urn Simulation ($10^{int(np.log10(num_steps))}$ steps)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Histogram
gen_hist = True
if gen_hist:
    total_trials = int(1e4)
    red_prop = np.zeros(total_trials)
    for trial in range(total_trials):
        red_prop[trial] = polyas_run_no_storage(num_steps, red, green, starting_balls)

    plt.figure(figsize=(8,5))
    plt.hist(red_prop, bins=20, density=True, color='red', edgecolor='black', alpha=0.7)

    # Mark the initial fraction
    plt.axvline(initial_fraction, color='black', linestyle='--', label='Initial Fraction')
    plt.text(initial_fraction + 0.01, plt.gca().get_ylim()[1]*0.9, f'Start={initial_fraction:.2f}', color='black')

    plt.xlabel("Fraction of Balls that are Red")
    plt.ylabel("Density")
    plt.title(rf"Distribution of Red Ball Fraction over $10^{int(np.log10(total_trials))}$ Trials ($10^{int(np.log10(num_steps))}$ steps)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()