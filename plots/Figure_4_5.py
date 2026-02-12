# %% [markdown]
# Figure 4-5 template
# - Anisotropic comparison across epsilon rows + reference row

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from plots.common import add_repo_to_syspath, find_repo_root, load_fdm_array, load_pickle, maybe_save_figure
    from plots.figure_blocks import STYLE_FIG4_5, apply_style, plot_ps_anisotropic_panel
except ModuleNotFoundError:
    from common import add_repo_to_syspath, find_repo_root, load_fdm_array, load_pickle, maybe_save_figure
    from figure_blocks import STYLE_FIG4_5, apply_style, plot_ps_anisotropic_panel


# %% [markdown]
# A. Paths / Config

# %%
ROOT = find_repo_root()
add_repo_to_syspath(ROOT)
apply_style(STYLE_FIG4_5)

FDM_FILE = None
fdm_data = load_fdm_array(FDM_FILE, ROOT)

PI_FILES_5D = [
    str(ROOT / "outputs/ps_run/5d/PI_N25000_it500_ep5000_epsilon0_seed0.pkl"),
    str(ROOT / "outputs/ps_run/5d/PI_N25000_it500_ep5000_epsilon01_seed0.pkl"),
    str(ROOT / "outputs/ps_run/5d/PI_N25000_it500_ep5000_epsilon03_seed0.pkl"),
    str(ROOT / "outputs/ps_run/5d/PI_N25000_it500_ep5000_epsilon05_seed0.pkl"),
]

PI_FILES_11D = [
    str(ROOT / "outputs/ps_run/11d/PI_N121000_it500_ep5000_epsilon0_seed0.pkl"),
    str(ROOT / "outputs/ps_run/11d/PI_N121000_it500_ep5000_epsilon01_seed0.pkl"),
    str(ROOT / "outputs/ps_run/11d/PI_N121000_it500_ep5000_epsilon03_seed0.pkl"),
    str(ROOT / "outputs/ps_run/11d/PI_N121000_it500_ep5000_epsilon05_seed0.pkl"),
]

EPSILON_LIST = [0, 0.1, 0.3, 0.5]
TIMES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

SAVE_5D_PNG = ROOT / "plots" / "figures" / "Figure_4.png"
SAVE_5D_PDF = ROOT / "plots" / "figures" / "Figure_4.pdf"
SAVE_11D_PNG = ROOT / "plots" / "figures" / "Figure_5.png"
SAVE_11D_PDF = ROOT / "plots" / "figures" / "Figure_5.pdf"


# %% [markdown]
# B. Load helpers

# %%
def load_params_list(paths):
    return [load_pickle(Path(p))["params"] for p in paths]


# %% [markdown]
# C. Figure 4 (5D)

# %%
params_5d = load_params_list(PI_FILES_5D)
fig4 = plot_ps_anisotropic_panel(
    params_5d,
    5,
    fdm_data,
    epsilon_list=EPSILON_LIST,
    times=TIMES,
    tf=0.5,
    radius=1.0,
)
maybe_save_figure(fig4, SAVE_5D_PNG, SAVE_5D_PDF, dpi=300)
plt.show()


# %% [markdown]
# D. Figure 5 (11D)

# %%
params_11d = load_params_list(PI_FILES_11D)
fig5 = plot_ps_anisotropic_panel(
    params_11d,
    11,
    fdm_data,
    epsilon_list=EPSILON_LIST,
    times=TIMES,
    tf=0.5,
    radius=1.0,
)
maybe_save_figure(fig5, SAVE_11D_PNG, SAVE_11D_PDF, dpi=300)
plt.show()

# %%
