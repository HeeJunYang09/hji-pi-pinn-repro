# %% [markdown]
# Figure 7 template
# - PI collocation scaling comparison (error curves + training-time bars)

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

try:
    from plots.common import add_repo_to_syspath, find_repo_root, maybe_save_figure, summarize_runs
    from plots.figure_blocks import (
        STYLE_FIG7_ERR,
        STYLE_FIG7_TIME,
        plot_k_error_curves,
        plot_k_training_times,
    )
except ModuleNotFoundError:
    from common import add_repo_to_syspath, find_repo_root, maybe_save_figure, summarize_runs
    from figure_blocks import (
        STYLE_FIG7_ERR,
        STYLE_FIG7_TIME,
        plot_k_error_curves,
        plot_k_training_times,
    )


# %% [markdown]
# A. Paths / Config

# %%
ROOT = find_repo_root()
add_repo_to_syspath(ROOT)

SEEDS = [0, 100, 200, 300, 400]
IT_PI, EP_PI = 500, 5000
EPSILON_PI = "0"

OUT3 = ROOT / "outputs" / "ps_run" / "3d"
OUT5 = ROOT / "outputs" / "ps_run" / "5d"

SAVE_3D_ERR_PNG = ROOT / "plots" / "figures" / "Figure_7_a.png"
SAVE_3D_ERR_PDF = ROOT / "plots" / "figures" / "Figure_7_a.pdf"
SAVE_3D_TIME_PNG = ROOT / "plots" / "figures" / "Figure_7_b.png"
SAVE_3D_TIME_PDF = ROOT / "plots" / "figures" / "Figure_7_b.pdf"
SAVE_5D_ERR_PNG = ROOT / "plots" / "figures" / "Figure_7_c.png"
SAVE_5D_ERR_PDF = ROOT / "plots" / "figures" / "Figure_7_c.pdf"
SAVE_5D_TIME_PNG = ROOT / "plots" / "figures" / "Figure_7_d.png"
SAVE_5D_TIME_PDF = ROOT / "plots" / "figures" / "Figure_7_d.pdf"

print("ROOT =", ROOT)
print("OUT3 exists?", OUT3.exists())
print("OUT5 exists?", OUT5.exists())


# %% [markdown]
# B. 3D (k=1,2,3)

# %%
ns3 = [3000, 9000, 27000]
k3 = np.array([1, 2, 3])
labels3 = {3000: r"$k=1$", 9000: r"$k=2$", 27000: r"$k=3$"}

pi_mean_3d = {}
pi_std_3d = {}
train_mean_3d = {}
for n_int in ns3:
    l2_mean, l2_std, tt_mean, _ = summarize_runs(
        OUT3, mode="pi", n_int=n_int, seeds=SEEDS, it=IT_PI, ep=EP_PI, epsilon=EPSILON_PI
    )
    pi_mean_3d[n_int] = l2_mean
    pi_std_3d[n_int] = l2_std
    train_mean_3d[n_int] = tt_mean
print({n_int: train_mean_3d[n_int] for n_int in ns3})

fig3_err, _ = plot_k_error_curves(ns3, pi_mean_3d, pi_std_3d, labels3, ylim=(8e-3, 1), style_dict=STYLE_FIG7_ERR)
maybe_save_figure(fig3_err, SAVE_3D_ERR_PNG, SAVE_3D_ERR_PDF, dpi=300)
plt.show()

fig3_time, _ = plot_k_training_times(k3, ns3, train_mean_3d, ylim=(0, 7500), style_dict=STYLE_FIG7_TIME)
maybe_save_figure(fig3_time, SAVE_3D_TIME_PNG, SAVE_3D_TIME_PDF, dpi=300)
plt.show()


# %% [markdown]
# C. 5D (k=1,2,3)

# %%
ns5 = [5000, 25000, 125000]
k5 = np.array([1, 2, 3])
labels5 = {5000: r"$k=1$", 25000: r"$k=2$", 125000: r"$k=3$"}

pi_mean_5d = {}
pi_std_5d = {}
train_mean_5d = {}
for n_int in ns5:
    l2_mean, l2_std, tt_mean, _ = summarize_runs(
        OUT5, mode="pi", n_int=n_int, seeds=SEEDS, it=IT_PI, ep=EP_PI, epsilon=EPSILON_PI
    )
    pi_mean_5d[n_int] = l2_mean
    pi_std_5d[n_int] = l2_std
    train_mean_5d[n_int] = tt_mean
print({n_int: train_mean_5d[n_int] for n_int in ns5})

fig5_err, _ = plot_k_error_curves(ns5, pi_mean_5d, pi_std_5d, labels5, ylim=(8e-3, 1), style_dict=STYLE_FIG7_ERR)
maybe_save_figure(fig5_err, SAVE_5D_ERR_PNG, SAVE_5D_ERR_PDF, dpi=300)
plt.show()

fig5_time, _ = plot_k_training_times(k5, ns5, train_mean_5d, ylim=(0, 48000), style_dict=STYLE_FIG7_TIME)
maybe_save_figure(fig5_time, SAVE_5D_TIME_PNG, SAVE_5D_TIME_PDF, dpi=300)
plt.show()

# %%
