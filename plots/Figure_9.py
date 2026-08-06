# %% [markdown]
# Figure 9
# DeepONet comparison summary for the revised moving-obstacle experiment.
#
# The paper figure is tracked as plots/figures/Figure_9.pdf/png. Full DeepONet
# checkpoints are intentionally not tracked because they are intermediate
# artifacts. This companion script records and visualizes the corresponding
# relative L2-error summary used in the revised table.

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from plots.common import find_repo_root, maybe_save_figure
except ModuleNotFoundError:
    from common import find_repo_root, maybe_save_figure


ROOT = find_repo_root(Path(__file__).resolve().parent)
METRICS_PATH = ROOT / "outputs" / "deeponet_summary" / "figure9_metrics.json"
SAVE_PNG = ROOT / "plots" / "figures" / "Figure_9_metrics.png"
SAVE_PDF = ROOT / "plots" / "figures" / "Figure_9_metrics.pdf"

with METRICS_PATH.open("r", encoding="utf-8") as f:
    metrics = json.load(f)

targets = metrics["targets"]
pi_mean = np.asarray(metrics["pi_pinn"]["mean"], dtype=float)
pi_std = np.asarray(metrics["pi_pinn"]["std"], dtype=float)
pi_best = np.asarray(metrics["pi_pinn"]["best"], dtype=float)
don = np.asarray(metrics["deeponet"]["single_run"], dtype=float)

print("Target location | PI mean | PI std | PI best | DeepONet")
for target, mean, std, best, don_val in zip(targets, pi_mean, pi_std, pi_best, don):
    print(f"{target:>13} | {mean:.3e} | {std:.3e} | {best:.3e} | {don_val:.3e}")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "text.usetex": False,
    }
)

x = np.arange(len(targets))
width = 0.32

fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.bar(x - width / 2, pi_mean, width, yerr=pi_std, capsize=3, label="PINN-PI mean")
ax.scatter(x - width / 2, pi_best, marker="*", s=90, color="black", zorder=3, label="PINN-PI best")
ax.bar(x + width / 2, don, width, label="DeepONet")
ax.set_yscale("log")
ax.set_ylabel(r"Relative $L^2$-error at $t=0$")
ax.set_xticks(x)
ax.set_xticklabels(targets, rotation=20, ha="right")
ax.legend(frameon=False)
ax.grid(True, axis="y", which="both", alpha=0.25)
fig.tight_layout()

maybe_save_figure(fig, SAVE_PNG, SAVE_PDF, dpi=300)
plt.show()

