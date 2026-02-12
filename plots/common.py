"""Shared plotting utilities for all Figure_* templates."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def find_repo_root(
    start: Optional[Path] = None,
    markers: Sequence[str] = ("requirements.txt", "README.md", "hji_pi_pinn"),
    max_depth: int = 10,
) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for _ in range(max_depth):
        if all((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Repo root not found. Run this from inside the repository.")


def add_repo_to_syspath(root: Path) -> None:
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def set_plot_style(kind: str = "paper") -> None:
    if kind == "paper":
        style = {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.labelsize": 28,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "figure.titlesize": 16,
            "lines.linewidth": 1.2,
            "text.usetex": False,
        }
    elif kind == "error_curve":
        style = {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.labelsize": 26,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "figure.titlesize": 15,
            "lines.linewidth": 1.2,
            "text.usetex": False,
        }
    else:
        raise ValueError(f"Unknown style kind: {kind}")
    plt.rcParams.update(style)


def load_pickle(path: Path):
    from hji_pi_pinn.core.io import load_pickle as _load_pickle

    return _load_pickle(path)


def run_path(
    out_dir: Path,
    mode: str,
    n_int: int,
    seed: int,
    *,
    it: Optional[int] = None,
    ep: Optional[int] = None,
    epsilon: Optional[str] = None,
) -> Path:
    if mode == "pi":
        if it is None or ep is None:
            raise ValueError("For mode='pi', both it and ep are required.")
        if epsilon is not None:
            return out_dir / f"PI_N{n_int}_it{it}_ep{ep}_epsilon{epsilon}_seed{seed}.pkl"
        return out_dir / f"PI_N{n_int}_it{it}_ep{ep}_seed{seed}.pkl"
    if mode == "direct":
        if ep is None:
            raise ValueError("For mode='direct', ep is required.")
        if epsilon is not None:
            return out_dir / f"Direct_N{n_int}_ep{ep}_epsilon{epsilon}_seed{seed}.pkl"
        return out_dir / f"Direct_N{n_int}_ep{ep}_seed{seed}.pkl"
    raise ValueError(f"Invalid mode='{mode}'. Expected 'pi' or 'direct'.")


def load_run(
    out_dir: Path,
    mode: str,
    n_int: int,
    seed: int,
    *,
    it: Optional[int] = None,
    ep: Optional[int] = None,
    epsilon: Optional[str] = None,
):
    path = run_path(out_dir, mode, n_int, seed, it=it, ep=ep, epsilon=epsilon)
    if not path.exists():
        raise FileNotFoundError(f"Missing run file: {path}")
    return load_pickle(path)


def summarize_runs(
    out_dir: Path,
    mode: str,
    n_int: int,
    seeds: Iterable[int],
    *,
    it: Optional[int] = None,
    ep: Optional[int] = None,
    epsilon: Optional[str] = None,
    metric_key: str = "l2_history",
):
    seeds = list(seeds)
    r0 = load_run(out_dir, mode, n_int, seeds[0], it=it, ep=ep, epsilon=epsilon)
    t_len = len(r0[metric_key])

    metric_mat = np.zeros((t_len, len(seeds)))
    train_times = np.zeros((len(seeds),))
    last_values = np.zeros((len(seeds),))

    for i, s in enumerate(seeds):
        r = load_run(out_dir, mode, n_int, s, it=it, ep=ep, epsilon=epsilon)
        vec = np.asarray(r[metric_key], dtype=float)
        if vec.shape[0] != t_len:
            raise ValueError(
                f"Inconsistent {metric_key} length at seed={s}: {vec.shape[0]} vs {t_len}"
            )
        metric_mat[:, i] = vec
        train_times[i] = float(r.get("train_time", np.nan))
        last_values[i] = float(vec[-1])

    return (
        np.mean(metric_mat, axis=1),
        np.std(metric_mat, axis=1),
        float(np.nanmean(train_times)),
        float(np.min(last_values)),
    )


def load_fdm_array(path: Optional[Path] = None, root: Optional[Path] = None) -> np.ndarray:
    if path is not None:
        f = Path(path)
    else:
        root = Path(root or find_repo_root())
        candidates = sorted((root / "data").glob("PS3D_*.npy"))
        if not candidates:
            raise FileNotFoundError("No FDM file found under data/PS3D_*.npy")
        f = candidates[0]
    if not f.exists():
        raise FileNotFoundError(f"FDM file not found: {f}")
    return np.load(f)


def compute_error_metrics(v_pred: np.ndarray, v_ref: np.ndarray, eps: float = 1e-12):
    err = np.asarray(v_pred) - np.asarray(v_ref)
    mse = float(np.mean(err**2))
    rel_l2 = float(np.linalg.norm(err.ravel()) / (np.linalg.norm(v_ref.ravel()) + eps))
    max_abs = float(np.max(np.abs(err)))
    return {"MSE": mse, "Relative L2": rel_l2, "Max Error": max_abs}


def make_xg(ss: int = 100, low: float = -0.5, high: float = 0.5):
    xg = np.linspace(low, high, 2 * ss + 1)
    x1, x2 = np.meshgrid(xg, xg, indexing="ij")
    return xg, x1, x2


def eval_nn_partial_diag(params, t_val: float, xg: np.ndarray, n_dim: int, tf: float = 0.5, r: float = 1.0):
    from hji_pi_pinn.problems.publisher_subscriber import eval_nn_partial_diag_slice

    return eval_nn_partial_diag_slice(params, t_val, xg, n_dim, tf=tf, r=r)


def ref_partial_diag(v3: np.ndarray, n_dim: int):
    from hji_pi_pinn.problems.publisher_subscriber import ref_from_3d_fdm_partial_diag

    return ref_from_3d_fdm_partial_diag(v3, n_dim)


def contour_value(
    ax,
    x1: np.ndarray,
    x2: np.ndarray,
    v_vals: np.ndarray,
    *,
    levels=20,
    cmap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    cf = ax.contourf(
        x1,
        x2,
        np.asarray(v_vals).T,
        levels=levels,
        linewidths=0,
        antialiased=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    return cf


def maybe_save_figure(fig, out_png: Optional[Path] = None, out_pdf: Optional[Path] = None, dpi: int = 300):
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", bbox_inches="tight", dpi=dpi)
    if out_pdf is not None:
        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=dpi)
