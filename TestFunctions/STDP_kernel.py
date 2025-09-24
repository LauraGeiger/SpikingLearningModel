import numpy as np
import matplotlib.pyplot as plt

def kernel(delta_t, A_pos=0.01, A_neg=0.05, center=-100.0, sigma=30.0):
    """
    Biphasic kernel for GC->PC plasticity:
    - negative peak at delta_t ~ center (e.g. -100 ms)
    - positive at large |delta_t|
    """
    return A_pos - A_neg * np.exp(-((delta_t - center) ** 2) / (2 * sigma ** 2))



dts = np.linspace(-300, 0, 301)
vals = [kernel(dt) for dt in dts]

plt.plot(dts, vals)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Δt (ms)")
plt.ylabel("Δw")
plt.show()

