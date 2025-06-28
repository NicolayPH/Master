import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

plt.rcParams.update({"font.family": "serif", "font.serif": ["Computer Modern"], "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10, "figure.dpi": 300})

x, y = make_moons(n_samples=1000, noise=0.15, random_state=42)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x[y == 0, 0], x[y == 0, 1], label="Class 0", s=15)
ax.scatter(x[y == 1, 0], x[y == 1, 1], label="Class 1", s=15)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend()
fig.tight_layout()
fig.savefig("Two-moons.pdf")
plt.show()
