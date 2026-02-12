"""Reusable plotting blocks for Figure scripts.

This module keeps Figure_*_template.py files thin:
- config/path setup in figure files
- plotting/math blocks here
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from jax import random, vmap
from matplotlib.lines import Line2D

try:
    from plots.common import compute_error_metrics, contour_value, eval_nn_partial_diag, ref_partial_diag
except ModuleNotFoundError:
    from common import compute_error_metrics, contour_value, eval_nn_partial_diag, ref_partial_diag


STYLE_FIG1_2 = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "text.usetex": False,
}

STYLE_FIG3 = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 23,
    "axes.labelsize": 32,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 16,
    "figure.titlesize": 20,
    "lines.linewidth": 1.2,
    "text.usetex": False,
}

STYLE_FIG4_5 = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 32,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 16,
    "figure.titlesize": 15,
    "lines.linewidth": 1.2,
    "text.usetex": False,
}

STYLE_FIG6 = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 32,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 15,
    "lines.linewidth": 1.2,
    "text.usetex": False,
}

STYLE_FIG7_ERR = {
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

STYLE_FIG7_TIME = {
    "font.family": "serif",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 15,
    "lines.linewidth": 1.2,
    "text.usetex": False,
}


def apply_style(style_dict: Dict[str, object]) -> None:
    plt.rcParams.update(style_dict)


def build_path_components(problem_cfg: Dict[str, object]):
    from hji_pi_pinn.problems.path_planning import PathPlanningConfig, build_path_planning_components

    cfg = PathPlanningConfig(**problem_cfg)
    return build_path_planning_components(cfg)


def value_function_from_params_at_t(v_single, params, t_val: float, x1_grid: np.ndarray, x2_grid: np.ndarray):
    num_x = len(x1_grid)
    x_flat = jnp.stack([x1_grid.flatten(), x2_grid.flatten()], axis=-1)
    t_fixed = jnp.full((x_flat.shape[0], 1), t_val)
    tx_flat = jnp.concatenate([t_fixed, x_flat], axis=1)
    v_vals = vmap(lambda tx: v_single(tx, params))(tx_flat)
    return np.asarray(v_vals.reshape((num_x, num_x)))


def plot_value_function_at_t(
    t_val: float,
    v_vals: np.ndarray,
    ax,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    *,
    levels=20,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    cf = ax.contourf(
        x1_grid,
        x2_grid,
        v_vals,
        levels=levels,
        linewidths=0,
        antialiased=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(fr"Value function $v(t={t_val:.1f}, x)$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    return cf


def simulate_trajectory_from_policy(
    params,
    v_single,
    policy_update_fn,
    x0: jnp.ndarray,
    *,
    sigma: jnp.ndarray,
    t_final: float = 1.0,
    dt: float = 0.01,
    seed: int = 0,
):
    num_steps = int(t_final / dt)
    ts = jnp.linspace(0.0, t_final, num_steps)
    traj = [np.array(x0).reshape(-1)]
    key = random.PRNGKey(seed)
    x = np.array(x0)

    def get_policy(t_val: float, x_val: np.ndarray):
        tx = jnp.concatenate([jnp.array([t_val]), jnp.asarray(x_val)])
        batch = {"grid_tx": tx[None, :]}
        policy = policy_update_fn(params, batch)
        alpha = policy["alpha"][0]
        beta = policy["beta"][0]
        return alpha, beta

    for i in range(num_steps - 1):
        t_val = float(ts[i])
        key, subkey = random.split(key)
        d_bt = random.normal(subkey, shape=(2,)) * np.sqrt(dt)
        alpha, beta = get_policy(t_val, x)
        dx = (alpha + beta) * dt + sigma @ d_bt
        x = x + np.asarray(dx)
        traj.append(x)

    return ts, jnp.stack(traj)


def simulate_trajectories(
    params,
    v_single,
    policy_update_fn,
    initial_points: Sequence[jnp.ndarray],
    *,
    sigma: jnp.ndarray,
    t_final: float = 1.0,
    dt: float = 0.01,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ts_list = []
    trajectories = []
    for i, x0 in enumerate(initial_points):
        ts_i, traj_i = simulate_trajectory_from_policy(
            params,
            v_single,
            policy_update_fn,
            x0,
            sigma=sigma,
            t_final=t_final,
            dt=dt,
            seed=i * 2 + 1,
        )
        ts_list.append(ts_i)
        trajectories.append(traj_i)
    return jnp.array(ts_list), jnp.array(trajectories)


def plot_trajectory_integrate(
    ts,
    traj,
    x_obs_func,
    initial_points,
    *,
    ax=None,
    x_goal=(0.9, 0.9),
    current_idx: Optional[int] = None,
    tail_len: int = 20,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    traj = np.array(traj)
    ts = np.array(ts)
    n_traj = len(ts)
    obs_path = np.array([x_obs_func(t) for t in ts[0]])
    colors = ["C0", "C1", "C2", "C4", "C5"]

    if current_idx is None:
        current_idx = len(ts[0]) - 1

    start_idx = max(current_idx - tail_len, 0)
    tail_idxs = np.arange(start_idx, current_idx + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for k in range(n_traj):
        x00 = f"{initial_points[k][0].item():.1f}"
        x01 = f"{initial_points[k][1].item():.1f}"
        traj_label = fr"$x(0) = ({x00}, {x01})$"
        ax.plot(traj[k, tail_idxs, 0], traj[k, tail_idxs, 1], "-", color=colors[k], linewidth=1.2, alpha=0.7)
        ax.plot(traj[k, current_idx, 0], traj[k, current_idx, 1], "o", color=colors[k], label=traj_label, markersize=5)

    ax.plot(obs_path[tail_idxs, 0], obs_path[tail_idxs, 1], "-", color="red", linewidth=2, alpha=0.7)
    ax.plot(obs_path[current_idx, 0], obs_path[current_idx, 1], "o", color="red", label="obstacle", markersize=7)
    ax.plot(*x_goal, "o", color="black", label="target", markersize=7)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Trajectory under Learned Policy")
    ax.set_aspect("equal")
    return ax


def plot_path_fig1_panel(
    params,
    v_single,
    v_ref_stack: np.ndarray,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    *,
    times: Sequence[float],
    force_last_col_dark: bool = True,
):
    fig, axs = plt.subplots(3, len(times), figsize=(24, 12))
    for i, t_val in enumerate(times):
        levels = 20
        v_pinn = value_function_from_params_at_t(v_single, params, t_val, x1_grid, x2_grid)
        v_ref_t = np.asarray(v_ref_stack[i])
        diff = v_ref_t - v_pinn

        err = compute_error_metrics(v_pinn, v_ref_t)
        rel_l2 = err["Relative L2"]
        y_label = r"$x_2$" if i == 0 else None

        cf_pinn = plot_value_function_at_t(t_val, v_pinn, axs[0][i], x1_grid, x2_grid, y_label=y_label, cmap="turbo")
        cb = fig.colorbar(cf_pinn, ax=axs[0][i])
        cb.set_ticks(np.linspace(*cf_pinn.get_clim(), num=5))
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        axs[0][i].set_title(fr"$v^{{PI}}_M(t,x;\theta_M)$ at $t$={t_val:.2f}")

        cf_fdm = plot_value_function_at_t(t_val, v_ref_t, axs[1][i], x1_grid, x2_grid, y_label=y_label, cmap="turbo")
        cb = fig.colorbar(cf_fdm, ax=axs[1][i])
        cb.set_ticks(np.linspace(*cf_fdm.get_clim(), num=5))
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        axs[1][i].set_title(fr"$v^R(t,x)$ at $t$={t_val:.2f}")

        if force_last_col_dark and i == len(times) - 1:
            levels = 0
        cf_diff = plot_value_function_at_t(
            t_val,
            np.abs(diff),
            axs[2][i],
            x1_grid,
            x2_grid,
            y_label=y_label,
            x_label=r"$x_1$",
            levels=levels,
            cmap="magma",
        )
        axs[2][i].set_title(fr"$\left|v^R(t,x) - v^{{PI}}_M(t,x;\theta_M)\right|$")
        cb = fig.colorbar(cf_diff, ax=axs[2][i])
        cb.set_ticks(np.linspace(*cf_diff.get_clim(), num=5))
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if force_last_col_dark and i == len(times) - 1:
            cf_diff.set_clim(1e-6, 1e-4)
            offset_text = cb.ax.yaxis.get_offset_text()
            offset_text.set_x(6.0)
            offset_text.set_y(-1)

        axs[2][i].text(
            0.5,
            -0.33,
            f"Relative $L^2$-error: {rel_l2:.2e}",
            transform=axs[2][i].transAxes,
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
        )

        for row in range(3):
            axs[row][i].set_xticks([-1, 0, 1])
            axs[row][i].set_yticks([-1, 0, 1])
            axs[row][i].set_rasterized(True)

    plt.subplots_adjust(wspace=-0.1, hspace=0.4, left=0.05, right=0.95, top=0.9, bottom=0.15)
    return fig


def plot_path_fig2_panel(
    params,
    v_single,
    ts_array,
    traj_array,
    x_obs_func,
    initial_points,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    *,
    time_indices: Sequence[int],
    time_values: Sequence[float],
):
    fig, ax = plt.subplots(1, len(time_values), figsize=(24, 8))

    for idx, (time_idx, t_val) in enumerate(zip(time_indices, time_values)):
        v_pinn = value_function_from_params_at_t(v_single, params, t_val, x1_grid, x2_grid)
        cf = ax[idx].contourf(x1_grid, x2_grid, v_pinn, levels=20, alpha=0.9, cmap="Greys")
        cbar = fig.colorbar(cf, ax=ax[idx], shrink=0.32)
        cbar.set_ticks(np.linspace(*cf.get_clim(), num=5))
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        y_label = r"$x_2$" if idx == 0 else None
        ax[idx] = plot_trajectory_integrate(
            ts_array,
            traj_array,
            x_obs_func,
            initial_points,
            ax=ax[idx],
            current_idx=time_idx,
            tail_len=30,
            y_label=y_label,
            x_label=r"$x_1$",
        )
        ax[idx].set_title(rf"$t = {t_val:.2f}$", fontsize=22)

        if idx == len(ax) - 1:
            handles, labels = ax[idx].get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax[idx].legend(
                unique.values(),
                unique.keys(),
                loc="upper left",
                bbox_to_anchor=(1.45, 1.1),
                borderaxespad=0.0,
                frameon=False,
            )

        ax[idx].set_rasterized(True)
        ax[idx].set_xticks([-1, 0, 1])
        ax[idx].set_yticks([-1, 0, 1])

    plt.tight_layout(h_pad=0.1)
    return fig


def plot_ps_fig3_panel(
    params_pi,
    params_direct,
    fdm_data: np.ndarray,
    *,
    n_dim: int,
    times: Sequence[float],
    tf: float = 0.5,
    radius: float = 1.0,
    force_last_col_dark: bool = True,
):
    xg = np.linspace(-0.5, 0.5, 201)
    x1, x2 = np.meshgrid(xg, xg, indexing="ij")

    fig = plt.figure(figsize=(42, 20))
    gs = gridspec.GridSpec(
        5,
        len(times),
        width_ratios=[1.1] * len(times),
        height_ratios=[1, 1, -0.1, 1, -0.02],
        wspace=0.3,
        hspace=0.55,
    )
    grid_rows = [0, 1, 3]
    axs = np.empty((3, len(times)), dtype=object)
    for r in range(3):
        for c in range(len(times)):
            axs[r, c] = fig.add_subplot(gs[grid_rows[r], c])

    for i, t_val in enumerate(times):
        v_pi = eval_nn_partial_diag(params_pi, t_val, xg, n_dim, tf=tf, r=radius)
        v_direct = eval_nn_partial_diag(params_direct, t_val, xg, n_dim, tf=tf, r=radius)
        v_ref = ref_partial_diag(fdm_data[i], n_dim)

        err_pi = np.abs(v_ref - v_pi)
        err_direct = np.abs(v_ref - v_direct)
        local_vmax = max(float(np.max(err_pi)), float(np.max(err_direct)))
        levels_err = np.linspace(0.0, local_vmax, 21) if local_vmax > 0 else 20
        is_last_col = i == len(times) - 1
        levels_err_plot = 0 if (force_last_col_dark and is_last_col) else levels_err

        x_label = r"$x_0$"
        y_label = r"$x_1=x_2$" if i == 0 else None

        cf_ref = contour_value(
            axs[0, i],
            x1,
            x2,
            v_ref,
            levels=20,
            cmap="turbo",
            title=fr"$v^R(t,x)$ at $t={t_val:.2f}$",
            x_label=x_label,
            y_label=y_label,
        )
        cb_ref = fig.colorbar(cf_ref, ax=axs[0, i])
        cb_ref.set_ticks(np.linspace(*cf_ref.get_clim(), num=5))
        cb_ref.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        m_pi = compute_error_metrics(v_pi, v_ref)
        cf_pi = contour_value(
            axs[1, i],
            x1,
            x2,
            err_pi,
            levels=levels_err_plot,
            cmap="magma",
            vmin=0.0,
            vmax=local_vmax,
            title=fr"$\left|v^R(t,x) - v^{{PI}}_M(t,x;\theta_M)\right|$",
            x_label=x_label,
            y_label=y_label,
        )
        cb_pi = fig.colorbar(cf_pi, ax=axs[1, i])
        cb_pi.set_ticks(np.linspace(0.0, local_vmax, num=5) if local_vmax > 0 else [0.0])
        cb_pi.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e" if local_vmax < 1e-3 else "%.3f"))
        axs[1, i].text(
            0.5,
            -0.30,
            f"Relative $L^2$-error: {m_pi['Relative L2']:.2e}",
            transform=axs[1, i].transAxes,
            ha="center",
            va="top",
            fontsize=22,
            fontweight="bold",
        )

        m_direct = compute_error_metrics(v_direct, v_ref)
        cf_direct = contour_value(
            axs[2, i],
            x1,
            x2,
            err_direct,
            levels=levels_err_plot,
            cmap="magma",
            vmin=0.0,
            vmax=local_vmax,
            title=fr"$\left|v^R(t,x) - v^D(t,x;\theta)\right|$",
            x_label=x_label,
            y_label=y_label,
        )
        cb_direct = fig.colorbar(cf_direct, ax=axs[2, i])
        cb_direct.set_ticks(np.linspace(0.0, local_vmax, num=5) if local_vmax > 0 else [0.0])
        cb_direct.ax.yaxis.set_major_formatter(
            mticker.FormatStrFormatter("%.1e" if local_vmax < 1e-3 else "%.3f")
        )
        if force_last_col_dark and is_last_col:
            cf_direct.set_clim(1e-10, 1e-4)
            cf_pi.set_clim(1e-10, 1e-4)
            off_direct = cb_direct.ax.yaxis.get_offset_text()
            off_direct.set_x(4)
            off_direct.set_y(-1)
            off_pi = cb_pi.ax.yaxis.get_offset_text()
            off_pi.set_x(4)
            off_pi.set_y(-1)

        axs[2, i].text(
            0.5,
            -0.30,
            f"Relative $L^2$-error: {m_direct['Relative L2']:.2e}",
            transform=axs[2, i].transAxes,
            ha="center",
            va="top",
            fontsize=22,
            fontweight="bold",
        )

        for r in range(3):
            axs[r, i].set_xticks([-0.5, 0.0, 0.5])
            axs[r, i].set_yticks([-0.5, 0.0, 0.5])
            axs[r, i].set_rasterized(True)

    return fig


def plot_ps_anisotropic_panel(
    params_pi_nd_list,
    n_dim: int,
    fdm_data: np.ndarray,
    *,
    epsilon_list: Optional[Sequence[float]] = None,
    times: Optional[Sequence[float]] = None,
    tf: float = 0.5,
    radius: float = 1.0,
):
    epsilon_list = list(epsilon_list) if epsilon_list is not None else [0, 0.1, 0.3, 0.5]
    times = list(times) if times is not None else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    xg = np.linspace(-0.5, 0.5, 201)
    x1, x2 = jnp.meshgrid(xg, xg, indexing="ij")

    fig = plt.figure(figsize=(31, 21))
    gs_main = gridspec.GridSpec(5, 6, width_ratios=[1, 1, 1, 1, 1, 1], wspace=0.4, hspace=0.3)

    axs = np.empty((5, 6), dtype=object)
    for i in range(5):
        for j in range(6):
            axs[i, j] = fig.add_subplot(gs_main[i, j])

    index_list = [0, 1, 2, 3]
    for i, t_val in enumerate(times):
        current_v_list: List[np.ndarray] = []
        col_vmax = -1e9
        col_vmin = 1e9

        for ei, _ in enumerate(index_list):
            params_pi_nd = params_pi_nd_list[len(index_list) - ei - 1]
            v_pi = eval_nn_partial_diag(params_pi_nd, t_val, xg, n_dim, tf=tf, r=radius)
            current_v_list.append(v_pi)
            col_vmax = max(col_vmax, float(np.max(v_pi)))
            col_vmin = min(col_vmin, float(np.min(v_pi)))

        v_fdm_3d = ref_partial_diag(fdm_data[i], n_dim)
        col_vmax = max(col_vmax, float(np.max(v_fdm_3d)))
        col_vmin = min(col_vmin, float(np.min(v_fdm_3d)))
        common_levels = np.linspace(col_vmin, col_vmax, 21)
        fmt = "%.1e" if abs(col_vmax) < 1e-2 else "%.2f"

        for ei, _ in enumerate(index_list):
            y_label = r"$x_i=x_j$" if i == 0 else None
            cf = contour_value(
                axs[ei][i],
                x1,
                x2,
                current_v_list[ei],
                levels=common_levels,
                cmap="turbo",
                vmin=col_vmin,
                vmax=col_vmax,
                y_label=y_label,
            )
            c_bar = fig.colorbar(cf, ax=axs[ei][i])
            c_bar.ax.set_ylim(col_vmin, col_vmax)
            c_bar.set_ticks(np.linspace(col_vmin, col_vmax, num=5))
            c_bar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
            axs[ei][i].set_rasterized(True)
            if ei == 0:
                axs[ei][i].set_title(fr"$v^{{PI}}_M(t,x;\theta_M)$ at $t$={t_val:.2f}", pad=20)
            else:
                axs[ei][i].set_title("")

        cf_ref = contour_value(
            axs[4][i],
            x1,
            x2,
            v_fdm_3d,
            levels=common_levels,
            cmap="turbo",
            vmin=col_vmin,
            vmax=col_vmax,
            x_label=r"$x_0$",
            y_label=(r"$x_i=x_j$" if i == 0 else None),
        )
        axs[4][i].set_title(fr"$v^R(t,x)$ at $t$={t_val:.2f}")
        axs[4][i].set_rasterized(True)
        c_bar_ref = fig.colorbar(cf_ref, ax=axs[4][i])
        c_bar_ref.ax.set_ylim(col_vmin, col_vmax)
        c_bar_ref.set_ticks(np.linspace(col_vmin, col_vmax, num=5))
        c_bar_ref.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))

        for k in range(5):
            axs[k][i].set_xticks([-0.5, 0, 0.5])
            axs[k][i].set_yticks([-0.5, 0, 0.5])

    row_labels = [rf"$\epsilon={e}$" for e in reversed(epsilon_list)] + [r"$\epsilon=0$"]
    fig.canvas.draw()
    for r in range(5):
        pos = axs[r, -1].get_position()
        x_text = pos.x1 + 0.09
        y_text = 0.5 * (pos.y0 + pos.y1)
        fig.text(x_text, y_text, row_labels[r], ha="right", va="center", fontsize=32, fontweight="bold")

    pos_upper = axs[3, 0].get_position()
    pos_lower = axs[4, 0].get_position()
    y_line = 0.5 * (pos_upper.y0 + pos_lower.y1)
    x_left = axs[0, 0].get_position().x0
    x_right = axs[0, 5].get_position().x1
    fig.add_artist(Line2D([x_left, x_right], [y_line, y_line], transform=fig.transFigure, linewidth=2.5, color="k"))
    return fig


def plot_rel_l2_compare_pi_vs_direct(pi_mean, pi_std, direct_mean, direct_std):
    x_axis = np.arange(len(pi_mean))
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogy(x_axis, pi_mean, label=r"Policy-iterative PINN")
    ax.semilogy(x_axis, direct_mean, label=r"Direct PINN")

    eps = 1e-12
    pi_lower = np.maximum(pi_mean - pi_std, eps)
    pi_upper = np.maximum(pi_mean + pi_std, eps)
    direct_lower = np.maximum(direct_mean - direct_std, eps)
    direct_upper = np.maximum(direct_mean + direct_std, eps)
    ax.fill_between(x_axis, pi_lower, pi_upper, alpha=0.2)
    ax.fill_between(x_axis, direct_lower, direct_upper, alpha=0.2)

    ax.legend()
    ax.set_xlim(0, len(x_axis) - 1)
    ax.set_ylim(8e-3, 1)
    ax.set_xlabel(r"Training epochs ($\times$5000)")
    ax.set_ylabel(r"Relative $L^2$-error")
    return fig, ax


def plot_k_error_curves(n_list, mean_dict, std_dict, labels, *, ylim=(8e-3, 1), style_dict=None):
    t_len = next(iter(mean_dict.values())).shape[0]
    x_axis = np.arange(t_len)
    style = STYLE_FIG7_ERR if style_dict is None else style_dict

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 6))
        eps = 1e-12
        for n_int in n_list:
            m = mean_dict[n_int]
            s = std_dict[n_int]
            ax.semilogy(x_axis, m, label=labels.get(n_int, f"N={n_int}"))
            lower = np.maximum(m - s, eps)
            upper = np.maximum(m + s, eps)
            ax.fill_between(x_axis, lower, upper, alpha=0.2)
        ax.legend()
        ax.set_xlim(0, t_len - 1)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"Training epochs ($\times$5000)")
        ax.set_ylabel(r"Relative $L^2$-error")
        return fig, ax


def plot_k_training_times(k_vals, n_list, train_mean_dict, *, ylim, style_dict=None):
    train_times = np.array([train_mean_dict[n_int] for n_int in n_list], dtype=float)
    train_times_round = np.rint(train_times).astype(int)
    style = STYLE_FIG7_TIME if style_dict is None else style_dict

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [f"C{i}" for i in range(len(n_list))]
        bars = ax.bar(k_vals, train_times, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([rf"$k={k}$" + "\n" + rf"$(N_{{\mathrm{{int}}}}={n})$" for k, n in zip(k_vals, n_list)])
        ax.set_xlabel(r"Collocation scaling ($k$)")
        ax.set_ylabel("Training time")
        ax.set_ylim(ylim)

        for rect, lab in zip(bars, train_times_round):
            h = rect.get_height()
            ax.annotate(
                f"{lab:d}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()
        return fig, ax
