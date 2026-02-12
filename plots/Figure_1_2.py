# %% [markdown]
# Figure 1-2 template
# - Figure 1: PI-PINN vs reference (value/error panels)
# - Figure 2: contour + trajectory snapshots under learned policy

# %%
import warnings
from pathlib import Path
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config

warnings.filterwarnings("ignore")
config.update("jax_default_matmul_precision", "float32")

try:
    from plots.common import add_repo_to_syspath, find_repo_root, maybe_save_figure
    from plots.figure_blocks import (
        STYLE_FIG1_2,
        apply_style,
        build_path_components,
        plot_path_fig1_panel,
        plot_path_fig2_panel,
        simulate_trajectories,
    )
except ModuleNotFoundError:
    from common import add_repo_to_syspath, find_repo_root, maybe_save_figure
    from figure_blocks import (
        STYLE_FIG1_2,
        apply_style,
        build_path_components,
        plot_path_fig1_panel,
        plot_path_fig2_panel,
        simulate_trajectories,
    )


# %% [markdown]
# A. Paths / Config

# %%
ROOT = find_repo_root()
add_repo_to_syspath(ROOT)
apply_style(STYLE_FIG1_2)

from hji_pi_pinn.core.models import count_params

problem_cfg = {
    "domain_t": [0.0, 1.0],
    "domain_x": [-1.0, 1.0],
    "dim_x": 2,
    "tf": 1.0,
    "lambda_1": 0.1,
    "lambda_2": 100.0,
    "lambda_3": 10.0,
    "delta": 0.1,
    "epsilon": 0.3,
    "x_goal": [0.9, 0.9],
    "sigma_scale": 0.1,
}

CKPT_PATH = ROOT / "outputs" / "path_run" / "PI_N4000_it1000_ep1000_seed0.pkl"
FDM_PATH = ROOT / "data" / "PP2D_bcN_l10p1_l2100_l310_om-2_2_nx101.npy"

SAVE_FIG1_PNG = ROOT / "plots" / "figures" / "Figure_1.png"
SAVE_FIG1_PDF = ROOT / "plots" / "figures" / "Figure_1.pdf"
SAVE_FIG2_PNG = ROOT / "plots" / "figures" / "Figure_2.png"
SAVE_FIG2_PDF = ROOT / "plots" / "figures" / "Figure_2.pdf"

print("ROOT =", ROOT)
print("CKPT exists?", CKPT_PATH.exists())
print("FDM exists?", FDM_PATH.exists())


# %% [markdown]
# B. Load model / reference + problem components

# %%
with open(CKPT_PATH, "rb") as f:
    load_data = pickle.load(f)
params = load_data["params"]
print("num_params =", count_params(params))

v_fdm = np.load(FDM_PATH)
print("v_fdm shape =", v_fdm.shape)

components = build_path_components(problem_cfg)
v_single = components["v_single"]
policy_update_fn = components["policy_update_fn"]
sigma = components["sigma"]


# %% [markdown]
# C. Figure 1

# %%
times = [0.0, 0.25, 0.5, 0.75, 1.0]
test_num_x = 101
x = np.linspace(-1, 1, test_num_x)
x1, x2 = np.meshgrid(x, x, indexing="ij")

fig1 = plot_path_fig1_panel(
    params,
    v_single,
    v_fdm,
    x1,
    x2,
    times=times,
    force_last_col_dark=True,
)
maybe_save_figure(fig1, SAVE_FIG1_PNG, SAVE_FIG1_PDF, dpi=300)
plt.show()


# %% [markdown]
# D. Figure 2

# %%
initial_points = [
    jnp.array([-0.6, -0.6]),
    jnp.array([-0.3, -0.3]),
    jnp.array([0.0, 0.0]),
    jnp.array([0.3, 0.3]),
    jnp.array([0.6, 0.6]),
]

def x_obs(t):
    return jnp.array([0.5 * jnp.cos(jnp.pi * t), 0.5 * jnp.sin(jnp.pi * t)])

ts_array, traj_array = simulate_trajectories(
    params,
    v_single,
    policy_update_fn,
    initial_points,
    sigma=sigma,
    t_final=1.0,
    dt=0.01,
)

fig2 = plot_path_fig2_panel(
    params,
    v_single,
    ts_array,
    traj_array,
    x_obs,
    initial_points,
    x1,
    x2,
    time_indices=[0, 24, 49, 74, 99],
    time_values=times,
)
maybe_save_figure(fig2, SAVE_FIG2_PNG, SAVE_FIG2_PDF, dpi=300)
plt.show()

# %%
