
#%%
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('kappa0p5_totEn_log.dat')

x = data[:, 0]
y = data[:, 1]

# Identify increasing intervals
increasing = np.diff(y) > 0

# Create figure
plt.figure(figsize=(8, 6))

# Plot the curve
plt.semilogx(x, y, 'b-', linewidth=1.5,
             label='y(x)')

# Highlight increasing intervals
for i in np.where(increasing)[0]:
    plt.semilogy(x[i:i+2], y[i:i+2],
                 'r-', linewidth=2.5)

# Labels and formatting
plt.xlabel(r'$x$', fontsize=15)
plt.ylabel(r'$y(x)$', fontsize=15)
plt.title('Logarithmic Y-axis', fontsize=15)

plt.grid(True, which='both', linestyle='--',
         alpha=0.4)

plt.legend(fontsize=12)

plt.tight_layout()

# Save figure
plt.savefig('kappa0p5_totEn_log.png',
            dpi=300, bbox_inches='tight')

plt.show()
#%%