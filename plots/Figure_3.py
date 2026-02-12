# %% [markdown]
# Figure 3 template
# - Reference vs PI vs Direct (partial-diagonal 2D slice)

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from plots.common import add_repo_to_syspath, find_repo_root, load_fdm_array, load_pickle, load_run, maybe_save_figure
    from plots.figure_blocks import STYLE_FIG3, apply_style, plot_ps_fig3_panel
except ModuleNotFoundError:
    from common import add_repo_to_syspath, find_repo_root, load_fdm_array, load_pickle, load_run, maybe_save_figure
    from figure_blocks import STYLE_FIG3, apply_style, plot_ps_fig3_panel


# %% [markdown]
# A. Paths / Config

# %%
ROOT = find_repo_root()
add_repo_to_syspath(ROOT)
apply_style(STYLE_FIG3)

OUT3 = ROOT / "outputs" / "ps_run" / "3d"
PI_SPEC = dict(mode="pi", n_int=9000, seed=400, it=500, ep=5000, epsilon=0)
DIRECT_SPEC = dict(mode="direct", n_int=9000, seed=100, ep=2500000, epsilon=0)

PI_FILE = None
DIRECT_FILE = None
FDM_FILE = None

N_DIM = 3
TF = 0.5
RADIUS = 1.0
TIMES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

SAVE_PNG = ROOT / "plots" / "figures" / "Figure_3.png"
SAVE_PDF = ROOT / "plots" / "figures" / "Figure_3.pdf"

print("ROOT:", ROOT)
print("OUT3 exists:", OUT3.exists())


# %% [markdown]
# B. Load params + reference

# %%
pi_run = load_pickle(Path(PI_FILE)) if PI_FILE is not None else load_run(OUT3, **PI_SPEC)
direct_run = load_pickle(Path(DIRECT_FILE)) if DIRECT_FILE is not None else load_run(OUT3, **DIRECT_SPEC)

params_pi = pi_run["params"]
params_direct = direct_run["params"]
fdm_data = load_fdm_array(FDM_FILE, ROOT)
print("FDM shape:", fdm_data.shape)


# %% [markdown]
# C. Plot

# %%
fig = plot_ps_fig3_panel(
    params_pi,
    params_direct,
    fdm_data,
    n_dim=N_DIM,
    times=TIMES,
    tf=TF,
    radius=RADIUS,
    force_last_col_dark=True,
)
maybe_save_figure(fig, SAVE_PNG, SAVE_PDF, dpi=300)
plt.show()

# %%
