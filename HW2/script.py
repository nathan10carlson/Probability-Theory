# run_poisson_multiple_times.py
import subprocess

for i in range(10):
    print(f"\n--- Run {i+1} ---")
    subprocess.run(["python", "Poisson.py"])