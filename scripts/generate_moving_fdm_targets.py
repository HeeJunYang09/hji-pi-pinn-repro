from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class MovingFDMConfig:
    lambda_1: float = 0.1
    lambda_2: float = 100.0
    lambda_3: float = 10.0
    delta: float = 0.1
    epsilon: float = 0.3
    sigma_scale: float = 0.1
    domain_t: tuple[float, float] = (0.0, 1.0)
    domain_x: tuple[float, float] = (-2.0, 2.0)
    target_domain_x: tuple[float, float] = (-1.0, 1.0)
    num_x: int = 201
    num_t: int = 40401
    boundary: str = "neumann"
    snapshot_times: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def as_pair(value) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def load_config(path: str | Path) -> tuple[MovingFDMConfig, list[tuple[float, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    prob = dict(raw.get("problem", {}))
    prob["domain_t"] = as_pair(prob.get("domain_t", (0.0, 1.0)))
    prob["domain_x"] = as_pair(prob.get("domain_x", (-2.0, 2.0)))
    prob["target_domain_x"] = as_pair(prob.get("target_domain_x", (-1.0, 1.0)))
    prob["snapshot_times"] = tuple(float(t) for t in prob.get("snapshot_times", (0.0, 0.25, 0.5, 0.75, 1.0)))
    cfg = MovingFDMConfig(**prob)

    target_cfg = raw.get("targets", {})
    train_values = [float(x) for x in target_cfg.get("train_values", [])]
    test_values = [float(x) for x in target_cfg.get("test_values", [])]
    targets = list(product(train_values, train_values)) + list(product(test_values, test_values))
    if not targets:
        targets = [tuple(map(float, y)) for y in target_cfg.get("points", [])]
    if not targets:
        raise ValueError("No FDM targets were provided in the config.")
    return cfg, targets


def apply_boundary_2d(u: np.ndarray, boundary: str) -> np.ndarray:
    u_bc = u.copy()
    if boundary == "neumann":
        u_bc[0, :] = u_bc[1, :]
        u_bc[-1, :] = u_bc[-2, :]
        u_bc[:, 0] = u_bc[:, 1]
        u_bc[:, -1] = u_bc[:, -2]
    elif boundary == "extrap":
        u_bc[0, :] = 2 * u_bc[1, :] - u_bc[2, :]
        u_bc[-1, :] = 2 * u_bc[-2, :] - u_bc[-3, :]
        u_bc[:, 0] = 2 * u_bc[:, 1] - u_bc[:, 2]
        u_bc[:, -1] = 2 * u_bc[:, -2] - u_bc[:, -3]
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")
    return u_bc


def terminal_cost(cfg: MovingFDMConfig, x1: np.ndarray, x2: np.ndarray, target: tuple[float, float]) -> np.ndarray:
    return cfg.lambda_3 * ((x1 - target[0]) ** 2 + (x2 - target[1]) ** 2)


def obstacle_phi(cfg: MovingFDMConfig, t: float, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    obs_x = 0.5 * np.cos(np.pi * t)
    obs_y = 0.5 * np.sin(np.pi * t)
    dist2 = (x1 - obs_x) ** 2 + (x2 - obs_y) ** 2
    return np.exp(-dist2 / (2.0 * cfg.epsilon**2))


def crop_to_target(cfg: MovingFDMConfig, value: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = int(np.argmin(np.abs(x_grid - cfg.target_domain_x[0])))
    right = int(np.argmin(np.abs(x_grid - cfg.target_domain_x[1])))
    return value[left : right + 1, left : right + 1], x_grid[left : right + 1]


def solve_hji_fdm_snapshots(cfg: MovingFDMConfig, target: tuple[float, float]):
    padded_nx = cfg.num_x + 2
    t_grid = np.linspace(cfg.domain_t[0], cfg.domain_t[1], cfg.num_t)
    x_grid = np.linspace(cfg.domain_x[0], cfg.domain_x[1], cfg.num_x)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    x_grid_pad = np.linspace(cfg.domain_x[0] - dx, cfg.domain_x[1] + dx, padded_nx)
    x1, x2 = np.meshgrid(x_grid_pad, x_grid_pad, indexing="ij")

    sigma2 = cfg.sigma_scale**2
    v_next = np.zeros((padded_nx, padded_nx))
    v_next[1:-1, 1:-1] = terminal_cost(cfg, x1[1:-1, 1:-1], x2[1:-1, 1:-1], target)
    v_next = apply_boundary_2d(v_next, cfg.boundary)

    snapshot_index = {int(round(t * (cfg.num_t - 1))): t for t in cfg.snapshot_times}
    snapshots = {}
    target_x = None

    if cfg.num_t - 1 in snapshot_index:
        cropped, target_x = crop_to_target(cfg, v_next[1:-1, 1:-1], x_grid)
        snapshots[snapshot_index[cfg.num_t - 1]] = cropped.astype(np.float32)

    for n in range(cfg.num_t - 2, -1, -1):
        t = float(t_grid[n])
        phi = obstacle_phi(cfg, t, x1, x2)

        dv_dx = np.zeros_like(v_next)
        dv_dy = np.zeros_like(v_next)
        dv_dx[1:-1, 1:-1] = (v_next[2:, 1:-1] - v_next[:-2, 1:-1]) / (2 * dx)
        dv_dy[1:-1, 1:-1] = (v_next[1:-1, 2:] - v_next[1:-1, :-2]) / (2 * dx)
        grad_norm = np.sqrt(dv_dx**2 + dv_dy**2)

        hamiltonian = np.zeros_like(grad_norm)
        small_grad = grad_norm <= 2 * cfg.lambda_1
        hamiltonian[small_grad] = (
            -(grad_norm[small_grad] ** 2) / (4 * cfg.lambda_1)
            + cfg.lambda_2 * phi[small_grad]
            + cfg.delta * grad_norm[small_grad]
        )
        hamiltonian[~small_grad] = (
            -grad_norm[~small_grad]
            + cfg.lambda_1
            + cfg.lambda_2 * phi[~small_grad]
            + cfg.delta * grad_norm[~small_grad]
        )

        d2v_xx = np.zeros_like(v_next)
        d2v_yy = np.zeros_like(v_next)
        d2v_xx[1:-1, :] = (v_next[2:, :] - 2 * v_next[1:-1, :] + v_next[:-2, :]) / dx**2
        d2v_yy[:, 1:-1] = (v_next[:, 2:] - 2 * v_next[:, 1:-1] + v_next[:, :-2]) / dx**2
        diffusion = sigma2 * (d2v_xx + d2v_yy)

        v_curr = v_next.copy()
        v_curr[1:-1, 1:-1] = v_next[1:-1, 1:-1] + dt * (
            hamiltonian[1:-1, 1:-1] + 0.5 * diffusion[1:-1, 1:-1]
        )
        v_next = apply_boundary_2d(v_curr, cfg.boundary)

        if n in snapshot_index:
            cropped, target_x = crop_to_target(cfg, v_next[1:-1, 1:-1], x_grid)
            snapshots[snapshot_index[n]] = cropped.astype(np.float32)

    times = np.array(sorted(snapshots.keys()), dtype=np.float32)
    values = np.stack([snapshots[float(t)] for t in times], axis=0)
    return times, target_x, values, dx, dt


def output_name(target: tuple[float, float], cfg: MovingFDMConfig, date: str) -> str:
    return (
        f"{date}_moving_obstacle_fdm_yg_{target[0]:.3f}_{target[1]:.3f}_"
        f"lam1_{cfg.lambda_1}_lam2_{cfg.lambda_2}_lam3_{cfg.lambda_3}_"
        f"sigma_01I_nx{cfg.num_x}_snapshots.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FDM reference snapshots for moving-obstacle target locations.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--date", type=str, default="20260501")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    cfg, targets = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        save_path = outdir / output_name(target, cfg, args.date)
        if args.skip_existing and save_path.exists():
            print(f"[skip] {save_path}")
            continue

        print(f"[solve] target={target}")
        times, x, values, dx, dt = solve_hji_fdm_snapshots(cfg, target)
        metadata = {
            **cfg.__dict__,
            "sigma": (cfg.sigma_scale * np.eye(2)).tolist(),
            "x_goal": list(target),
            "dx": float(dx),
            "dt": float(dt),
        }
        np.savez_compressed(save_path, times=times, x=x, values=values, config=json.dumps(metadata))
        print(f"[saved] {save_path} times={times.tolist()} values.shape={values.shape}")


if __name__ == "__main__":
    main()
