# %% [markdown]
# Figure 6 template
# - PI vs Direct relative L2 error curves for 3D / 5D / 11D

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from plots.common import add_repo_to_syspath, find_repo_root, maybe_save_figure, summarize_runs
    from plots.figure_blocks import STYLE_FIG6, apply_style, plot_rel_l2_compare_pi_vs_direct
except ModuleNotFoundError:
    from common import add_repo_to_syspath, find_repo_root, maybe_save_figure, summarize_runs
    from figure_blocks import STYLE_FIG6, apply_style, plot_rel_l2_compare_pi_vs_direct


# %% [markdown]
# A. Paths / Config

# %%
ROOT = find_repo_root()
add_repo_to_syspath(ROOT)
apply_style(STYLE_FIG6)

SEEDS = [0, 100, 200, 300, 400]
IT_PI, EP_PI = 500, 5000
EP_DIRECT = 2500000
EPSILON_PI = "0"
EPSILON_DIRECT = "0"

OUT3 = ROOT / "outputs" / "ps_run" / "3d"
OUT5 = ROOT / "outputs" / "ps_run" / "5d"
OUT11 = ROOT / "outputs" / "ps_run" / "11d"

SAVE_3D_PNG = ROOT / "plots" / "figures" / "Figure_6_a.png"
SAVE_3D_PDF = ROOT / "plots" / "figures" / "Figure_6_a.pdf"
SAVE_5D_PNG = ROOT / "plots" / "figures" / "Figure_6_b.png"
SAVE_5D_PDF = ROOT / "plots" / "figures" / "Figure_6_b.pdf"
SAVE_11D_PNG = ROOT / "plots" / "figures" / "Figure_6_c.png"
SAVE_11D_PDF = ROOT / "plots" / "figures" / "Figure_6_c.pdf"

print("ROOT =", ROOT)
print("OUT3 exists?", OUT3.exists())
print("OUT5 exists?", OUT5.exists())
print("OUT11 exists?", OUT11.exists())


# %% [markdown]
# B. 3D

# %%
n3 = 9000
pi_mean_3d, pi_std_3d, pi_tt_3d, pi_min_3d = summarize_runs(
    OUT3, mode="pi", n_int=n3, seeds=SEEDS, it=IT_PI, ep=EP_PI, epsilon=EPSILON_PI
)
direct_mean_3d, direct_std_3d, direct_tt_3d, direct_min_3d = summarize_runs(
    OUT3, mode="direct", n_int=n3, seeds=SEEDS, ep=EP_DIRECT, epsilon=EPSILON_DIRECT
)

print("3D PI last:", pi_mean_3d[-1], "Direct last:", direct_mean_3d[-1])
print("3D PI min:", pi_min_3d, "Direct min:", direct_min_3d)
print("3D PI time(avg):", pi_tt_3d, "Direct time(avg):", direct_tt_3d)

fig3d, _ = plot_rel_l2_compare_pi_vs_direct(pi_mean_3d, pi_std_3d, direct_mean_3d, direct_std_3d)
maybe_save_figure(fig3d, SAVE_3D_PNG, SAVE_3D_PDF, dpi=300)
plt.show()


# %% [markdown]
# C. 5D

# %%
n5 = 25000
pi_mean_5d, pi_std_5d, pi_tt_5d, pi_min_5d = summarize_runs(
    OUT5, mode="pi", n_int=n5, seeds=SEEDS, it=IT_PI, ep=EP_PI, epsilon=EPSILON_PI
)
direct_mean_5d, direct_std_5d, direct_tt_5d, direct_min_5d = summarize_runs(
    OUT5, mode="direct", n_int=n5, seeds=SEEDS, ep=EP_DIRECT, epsilon=EPSILON_DIRECT
)

print("5D PI last:", pi_mean_5d[-1], "Direct last:", direct_mean_5d[-1])
print("5D PI min:", pi_min_5d, "Direct min:", direct_min_5d)
print("5D PI time(avg):", pi_tt_5d, "Direct time(avg):", direct_tt_5d)

fig5d, _ = plot_rel_l2_compare_pi_vs_direct(pi_mean_5d, pi_std_5d, direct_mean_5d, direct_std_5d)
maybe_save_figure(fig5d, SAVE_5D_PNG, SAVE_5D_PDF, dpi=300)
plt.show()


# %% [markdown]
# D. 11D

# %%
n11 = 121000
pi_mean_11d, pi_std_11d, pi_tt_11d, pi_min_11d = summarize_runs(
    OUT11, mode="pi", n_int=n11, seeds=SEEDS, it=IT_PI, ep=EP_PI, epsilon=EPSILON_PI
)
direct_mean_11d, direct_std_11d, direct_tt_11d, direct_min_11d = summarize_runs(
    OUT11, mode="direct", n_int=n11, seeds=SEEDS, ep=EP_DIRECT, epsilon=EPSILON_DIRECT
)

print("11D PI last:", pi_mean_11d[-1], "Direct last:", direct_mean_11d[-1])
print("11D PI min:", pi_min_11d, "Direct min:", direct_min_11d)
print("11D PI time(avg):", pi_tt_11d, "Direct time(avg):", direct_tt_11d)

fig11d, _ = plot_rel_l2_compare_pi_vs_direct(pi_mean_11d, pi_std_11d, direct_mean_11d, direct_std_11d)
maybe_save_figure(fig11d, SAVE_11D_PNG, SAVE_11D_PDF, dpi=300)
plt.show()

# %%
