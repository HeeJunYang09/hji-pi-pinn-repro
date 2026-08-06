# %% [markdown]
# Figure 8
# Small-diffusion publisher-subscriber comparison.
# This script recreates the 2x5 value-profile panel from tracked 3D checkpoints.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from plots.common import (
        add_repo_to_syspath,
        contour_value,
        eval_nn_partial_diag,
        find_repo_root,
        load_pickle,
        make_xg,
        maybe_save_figure,
    )
except ModuleNotFoundError:
    from common import (
        add_repo_to_syspath,
        contour_value,
        eval_nn_partial_diag,
        find_repo_root,
        load_pickle,
        make_xg,
        maybe_save_figure,
    )


ROOT = find_repo_root(Path(__file__).resolve().parent)
add_repo_to_syspath(ROOT)

DATA_DIR = ROOT / "outputs" / "ps_small_diffusion" / "3d"
SAVE_PNG = ROOT / "plots" / "figures" / "Figure_8.png"
SAVE_PDF = ROOT / "plots" / "figures" / "Figure_8.pdf"

GAMMAS = [
    ("0.10", "sigma01"),
    ("0.07", "sigma007"),
    ("0.05", "sigma005"),
    ("0.03", "sigma003"),
    ("0.01", "sigma001"),
]

# Best/checkpoint seeds used in the revised paper figure.
BEST_SEEDS = {
    "PI": {"sigma01": 4, "sigma007": 4, "sigma005": 1, "sigma003": 3, "sigma001": 3},
    "Direct": {"sigma01": 4, "sigma007": 1, "sigma005": 2, "sigma003": 2, "sigma001": 1},
}


def params_from_run(path: Path):
    run = load_pickle(path)
    if isinstance(run, dict):
        for key in ("params", "param", "net_params", "model_params"):
            if key in run:
                return run[key]
    raise KeyError(f"Could not find network parameters in {path}")


def find_checkpoint(method: str, sigma_token: str, seed: int) -> Path:
    if method == "PI":
        pattern = f"save_data_20260508_N9000_iter500_epoch5000_*_3D_{sigma_token}_seed{seed}.pkl"
    elif method == "Direct":
        pattern = f"save_data_20260508_Direct_N9000_epoch5000_*_3D_{sigma_token}_seed{seed}.pkl"
    else:
        raise ValueError(method)

    matches = sorted(DATA_DIR.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one checkpoint for {method}, {sigma_token}, seed={seed}; found {len(matches)}")
    return matches[0]


def load_profile(method: str, sigma_token: str) -> np.ndarray:
    seed = BEST_SEEDS[method][sigma_token]
    params = params_from_run(find_checkpoint(method, sigma_token, seed))
    return np.asarray(eval_nn_partial_diag(params, 0.0, XG, n_dim=3, tf=0.5, r=1.0))


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.usetex": False,
    }
)

XG, X1, X2 = make_xg(ss=100, low=-0.5, high=0.5)
profiles = {(method, token): load_profile(method, token) for method in ("PI", "Direct") for _, token in GAMMAS}
vmin = min(float(v.min()) for v in profiles.values())
vmax = max(float(v.max()) for v in profiles.values())

fig, axes = plt.subplots(2, len(GAMMAS), figsize=(22, 8.2), constrained_layout=True)
last_cf = None

for row, method in enumerate(("PI", "Direct")):
    for col, (gamma, token) in enumerate(GAMMAS):
        ax = axes[row, col]
        last_cf = contour_value(
            ax,
            X1,
            X2,
            profiles[(method, token)],
            levels=24,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            title=rf"$\gamma={gamma}$" if row == 0 else None,
            x_label=r"$x_0$",
            y_label=r"$s$" if col == 0 else None,
        )
        if col == 0:
            ax.text(
                -0.26,
                0.5,
                "PINN-PI" if method == "PI" else "Direct PINN",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=15,
            )

if last_cf is not None:
    fig.colorbar(last_cf, ax=axes.ravel().tolist(), shrink=0.92, pad=0.015)

maybe_save_figure(fig, SAVE_PNG, SAVE_PDF, dpi=300)
plt.show()

