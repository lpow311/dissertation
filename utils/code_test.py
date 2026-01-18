import numpy as np
import matplotlib.pyplot as plt

# Most calm, some moderately panicked, few highly panicked
panic_levels = np.random.beta(a=2, b=5, size=1000)

plt.hist(panic_levels, bins=20)
plt.xlabel("Panic Level")
plt.ylabel("Number of Agents")
plt.title("Distribution of Agent Panic")
plt.show()
