
#%%
import numpy as np
import matplotlib.pyplot as plt

# Load data from .dat file
data = np.loadtxt('kappa0p5_totEn.dat')

# Extract x and y columns
x = data[:, 0]
y = data[:, 1]

# Identify intervals where y increases
increasing = np.diff(y) > 0

# Plot
plt.figure(figsize=(8, 6))

plt.plot(x, y, 'b-', linewidth=1.5, label="$E_{\kappa}(t)$")

# Highlight increasing intervals
for i in np.where(increasing)[0]:
    plt.plot(x[i:i+2], y[i:i+2],
             'r-', linewidth=2.5)

plt.xlabel('t', fontsize=14)
plt.ylabel("$E_{\kappa}(t)$", fontsize=14)
plt.title("$\kappa$-entropy dissipation after T=1", fontsize=15)

plt.legend(fontsize=12)
#plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

# Save figure
plt.savefig('kappa0p5_totEn_plot_new.png', dpi=300)

plt.show()

# Print increasing intervals
print("Intervals where y increases:")

for i in np.where(increasing)[0]:
    print(f"x = {x[i]:.6f} to {x[i+1]:.6f}, "
          f"y = {y[i]:.6f} to {y[i+1]:.6f}")
#%%




