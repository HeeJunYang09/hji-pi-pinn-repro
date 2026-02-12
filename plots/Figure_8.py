# %% [markdown]
# Figure 8
# - (a) Path-planning Hamiltonian surface in p-space
# - (b) Publisher-subscriber Hamiltonian surface on (p2, p3) slice

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

try:
    from plots.common import find_repo_root, maybe_save_figure
except ModuleNotFoundError:
    from common import find_repo_root, maybe_save_figure


STYLE_FIG8 = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "figure.titlesize": 15,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "text.usetex": False,
}


def apply_style():
    plt.rcParams.update(STYLE_FIG8)


def phi_obstacle_np(t, x, epsilon=0.3):
    x_obs = np.array([0.5 * np.cos(np.pi * t), 0.5 * np.sin(np.pi * t)])
    return np.exp(-np.linalg.norm(x - x_obs) ** 2 / (2 * epsilon**2))


def h_path_planning(t, x, p, *, lambda_1, lambda_2, delta, epsilon):
    p_norm = np.linalg.norm(p)
    phi_val = phi_obstacle_np(t, x, epsilon=epsilon)
    if p_norm <= 2.0 * lambda_1:
        return -(1.0 / (4.0 * lambda_1)) * (p_norm**2) + lambda_2 * phi_val + delta * p_norm
    return -p_norm + lambda_1 + lambda_2 * phi_val + delta * p_norm


def plot_fig8_a():
    lambda_1 = 0.1
    lambda_2 = 100
    epsilon = 0.3
    delta = 0.1
    t_star = 0.0
    x_star = np.array([0.0, 0.0])
    p_max = 0.03
    num_points = 200

    p1 = np.linspace(-p_max, p_max, num_points)
    p2 = np.linspace(-p_max, p_max, num_points)
    p1g, p2g = np.meshgrid(p1, p2)
    h_vals = np.zeros_like(p1g)

    for i in range(num_points):
        for j in range(num_points):
            p_vec = np.array([p1g[i, j], p2g[i, j]])
            h_vals[i, j] = h_path_planning(
                t_star,
                x_star,
                p_vec,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                delta=delta,
                epsilon=epsilon,
            )

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    surf = ax.plot_surface(p1g, p2g, h_vals, cmap="plasma", alpha=0.9, linewidth=0)

    z_min, z_max = h_vals.min(), h_vals.max()
    ax.set_xlabel(r"$p_1$")
    ax.set_ylabel(r"$p_2$")
    ax.set_zlabel(r"$H(x, p)$", labelpad=15)
    ax.set_zticks([np.round(z_min, 4), np.round(z_max, 4)])
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.15, format="%.4f")
    ax.set_xticks([-p_max, 0, p_max])
    ax.set_yticks([-p_max, 0, p_max])
    ax.zaxis.get_offset_text().set_visible(False)
    ax.tick_params(axis="z", which="major", pad=15)
    ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.tight_layout(h_pad=0.1)
    return fig


def psi(x, *, alpha, beta):
    x = np.asarray(x)
    x0 = x[0]
    x_sq = x * x
    base = np.array([alpha * np.sin(x0), -beta * x0, -beta * x0])
    return base * x_sq


def h_publisher(x, p, *, A, B, C, alpha, beta):
    x = np.asarray(x)
    p = np.asarray(p)

    drift = A @ x + psi(x, alpha=alpha, beta=beta)
    term1 = p @ drift
    term2 = -np.sum(np.abs(B.T @ p))
    term3 = np.sum(np.abs(C.T @ p))
    return term1 + term2 + term3


def plot_fig8_b():
    n = 3
    a = 1.0
    b = 1.0
    c = 0.5
    alpha = -2.0
    beta = 2.0
    theta = np.pi / 4

    e1 = np.array([1.0, 0.0, 0.0])
    e = np.ones(n)
    A = np.outer(e1, e1) - (1.0 / n) * np.outer(e, e) + a * np.eye(n)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    B = np.zeros((n, n - 1))
    C = np.zeros((n, n - 1))
    B[1:, :] = b * np.eye(n - 1)
    C[1:, :] = c * R

    x_star = np.array([0.0, 0.0, 0.0])
    p_max = 2.0
    num_points = 200

    p2 = np.linspace(-p_max, p_max, num_points)
    p3 = np.linspace(-p_max, p_max, num_points)
    p2g, p3g = np.meshgrid(p2, p3)
    h_23 = np.zeros_like(p2g)

    for i in range(num_points):
        for j in range(num_points):
            p_vec = np.array([0.0, p2g[i, j], p3g[i, j]])
            h_23[i, j] = h_publisher(x_star, p_vec, A=A, B=B, C=C, alpha=alpha, beta=beta)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(p2g, p3g, h_23, cmap="plasma", linewidth=0, antialiased=True)
    ax.set_xlabel(r"$p_2$")
    ax.set_ylabel(r"$p_3$")
    ax.set_zlabel(r"$H(x, p)$", labelpad=10)
    ax.set_xticks([-2.0, -1.0, 0.0, 1.0, 2.0])
    ax.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    plt.tight_layout(h_pad=0.1)
    return fig


# %% [markdown]
# A. Paths / Config

# %%
ROOT = find_repo_root(Path(__file__).resolve().parent)
apply_style()

SAVE_A_PNG = ROOT / "plots" / "figures" / "Figure_8_a.png"
SAVE_A_PDF = ROOT / "plots" / "figures" / "Figure_8_a.pdf"
SAVE_B_PNG = ROOT / "plots" / "figures" / "Figure_8_b.png"
SAVE_B_PDF = ROOT / "plots" / "figures" / "Figure_8_b.pdf"

print("ROOT =", ROOT)


# %% [markdown]
# B. Figure 8(a)

# %%
fig_a = plot_fig8_a()
maybe_save_figure(fig_a, SAVE_A_PNG, SAVE_A_PDF, dpi=300)
plt.show()


# %% [markdown]
# C. Figure 8(b)

# %%
fig_b = plot_fig8_b()
maybe_save_figure(fig_b, SAVE_B_PNG, SAVE_B_PDF, dpi=300)
plt.show()

# %%
