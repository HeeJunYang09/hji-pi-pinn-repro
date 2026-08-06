from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax import grad, hessian, jit, random, value_and_grad, vmap
from jax.nn import initializers
from tqdm import tqdm

from hji_pi_pinn.core.io import save_pickle

jax.config.update("jax_default_matmul_precision", "highest")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_sensor_points(num_per_dim: int, domain_x: tuple[float, float]) -> jnp.ndarray:
    x = jnp.linspace(domain_x[0], domain_x[1], num_per_dim)
    x1, x2 = jnp.meshgrid(x, x, indexing="ij")
    return jnp.stack([x1.reshape(-1), x2.reshape(-1)], axis=-1)


def init_mlp(layer_dims: list[int], key):
    initializer = initializers.glorot_uniform()
    keys = random.split(key, len(layer_dims) - 1)
    return [
        (initializer(k, (in_dim, out_dim), jnp.float32), jnp.zeros((out_dim,), dtype=jnp.float32))
        for in_dim, out_dim, k in zip(layer_dims[:-1], layer_dims[1:], keys)
    ]


def apply_mlp(params, x):
    for w, b in params[:-1]:
        x = jnp.sin(jnp.matmul(x, w) + b)
    w, b = params[-1]
    return jnp.matmul(x, w) + b


def init_deeponet(branch_layers: list[int], trunk_layers: list[int], key):
    key_branch, key_trunk = random.split(key, 2)
    if branch_layers[-1] != trunk_layers[-1]:
        raise ValueError("Branch and trunk output dimensions must match.")
    return {
        "branch": init_mlp(branch_layers, key_branch),
        "trunk": init_mlp(trunk_layers, key_trunk),
        "bias": jnp.array(0.0, dtype=jnp.float32),
    }


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def make_problem_functions(prob: dict, sensor_points: jnp.ndarray):
    lambda_1 = float(prob["lambda_1"])
    lambda_2 = float(prob["lambda_2"])
    lambda_3 = float(prob["lambda_3"])
    delta = float(prob["delta"])
    epsilon = float(prob["epsilon"])
    sigma_scale = float(prob["sigma_scale"])
    sigma = sigma_scale * jnp.eye(2)

    def terminal_cost_x(x, y):
        return lambda_3 * jnp.sum((x - y) ** 2)

    def terminal_sensor_values(y):
        return vmap(lambda x: terminal_cost_x(x, y))(sensor_points)

    def deeponet_raw(params, y, tx):
        branch_input = terminal_sensor_values(y)
        branch = apply_mlp(params["branch"], branch_input)
        trunk = apply_mlp(params["trunk"], tx)
        return jnp.dot(branch, trunk) + params["bias"]

    def value_from_y_tx(params, y, tx):
        t, x1, x2 = tx
        x = jnp.array([x1, x2])
        return (1.0 - t) * deeponet_raw(params, y, tx) + terminal_cost_x(x, y)

    def obstacle_phi(t, x):
        obs = jnp.array([0.5 * jnp.cos(jnp.pi * t), 0.5 * jnp.sin(jnp.pi * t)])
        return jnp.exp(-jnp.sum((x - obs) ** 2) / (2.0 * epsilon**2))

    def spatial_hessian(params, y, t, x):
        def value_at_x(x_local):
            return value_from_y_tx(params, y, jnp.array([t, x_local[0], x_local[1]]))

        return hessian(value_at_x)(x)

    def find_policy_single(params, z):
        y = z[0:2]
        tx = z[2:5]
        grad_tx = grad(lambda tx_: value_from_y_tx(params, y, tx_))(tx)
        dv_dx = grad_tx[1:3]
        grad_norm = jnp.linalg.norm(dv_dx)
        alpha_unclipped = -dv_dx / (2.0 * lambda_1)
        alpha = jnp.where(grad_norm <= 2.0 * lambda_1, alpha_unclipped, -dv_dx / (grad_norm + 1e-10))
        beta = delta * dv_dx / (grad_norm + 1e-10)
        return alpha, beta

    def residual_single(params, z, alpha, beta):
        y = z[0:2]
        t, x1, x2 = z[2], z[3], z[4]
        x = jnp.array([x1, x2])
        tx = jnp.array([t, x1, x2])
        grad_tx = grad(lambda tx_: value_from_y_tx(params, y, tx_))(tx)
        dv_dt = grad_tx[0]
        dv_dx = grad_tx[1:3]
        diffusion = jnp.trace(sigma @ sigma.T @ spatial_hessian(params, y, t, x))
        residual = (
            dv_dt
            + lambda_1 * jnp.sum(alpha**2)
            + lambda_2 * obstacle_phi(t, x)
            + jnp.dot(dv_dx, alpha + beta)
            + 0.5 * diffusion
        )
        return residual**2

    return value_from_y_tx, find_policy_single, residual_single


def make_sampler(y_train: jnp.ndarray, prob: dict):
    domain_t = tuple(float(x) for x in prob["domain_t"])
    domain_x = tuple(float(x) for x in prob["domain_x"])

    def sample_training_batch(key, points_per_target: int):
        total_points = y_train.shape[0] * points_per_target
        _, key_t, key_x1, key_x2 = random.split(key, 4)
        y = jnp.repeat(y_train, points_per_target, axis=0)
        t = random.uniform(key_t, (total_points,), minval=domain_t[0], maxval=domain_t[1], dtype=jnp.float32)
        x1 = random.uniform(key_x1, (total_points,), minval=domain_x[0], maxval=domain_x[1], dtype=jnp.float32)
        x2 = random.uniform(key_x2, (total_points,), minval=domain_x[0], maxval=domain_x[1], dtype=jnp.float32)
        return jnp.concatenate([y, t[:, None], x1[:, None], x2[:, None]], axis=1)

    return sample_training_batch


def fdm_path_for_y(fdm_dir: Path, date: str, y) -> Path:
    return (
        fdm_dir
        / (
            f"{date}_moving_obstacle_fdm_yg_{float(y[0]):.3f}_{float(y[1]):.3f}_"
            "lam1_0.1_lam2_100.0_lam3_10.0_sigma_01I_nx201_snapshots.npz"
        )
    )


def relative_l2(pred, ref) -> float:
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))


def make_evaluator(value_from_y_tx, fdm_dir: Path, date: str):
    @jit
    def value_batch(params, z_batch):
        return vmap(lambda z: value_from_y_tx(params, z[0:2], z[2:5]))(z_batch)

    def value_grid_at_t(params, y, t, x_grid):
        x1, x2 = jnp.meshgrid(jnp.asarray(x_grid), jnp.asarray(x_grid), indexing="ij")
        z = jnp.stack(
            [
                jnp.full(x1.size, float(y[0]), dtype=jnp.float32),
                jnp.full(x1.size, float(y[1]), dtype=jnp.float32),
                jnp.full(x1.size, float(t), dtype=jnp.float32),
                x1.reshape(-1),
                x2.reshape(-1),
            ],
            axis=-1,
        )
        return np.asarray(value_batch(params, z).reshape(x1.shape))

    def evaluate_y(params, y):
        data = np.load(fdm_path_for_y(fdm_dir, date, y))
        times = data["times"]
        x_grid = data["x"]
        fdm_values = data["values"]
        errors = []
        for i, t in enumerate(times):
            pred = value_grid_at_t(params, y, float(t), x_grid)
            errors.append(relative_l2(pred, fdm_values[i]))
        return {"y": np.asarray(y), "errors": np.asarray(errors), "t0_error": float(errors[0]), "mean_error": float(np.mean(errors))}

    return evaluate_y


def format_target_label(y) -> str:
    y = np.asarray(y)
    return f"({float(y[0]):.2f},{float(y[1]):.2f})"


def train_one_seed(cfg: dict, seed: int, outdir: Path, fdm_dir: Path, date: str) -> Path:
    prob = cfg["problem"]
    trn = cfg["train"]
    don = cfg["deeponet"]
    y_train = jnp.asarray(cfg["targets"]["train"], dtype=jnp.float32)
    y_test = jnp.asarray(cfg["targets"]["test"], dtype=jnp.float32)

    sensor_points = make_sensor_points(int(don["sensor_num_per_dim"]), tuple(float(x) for x in prob["domain_x"]))
    value_from_y_tx, find_policy_single, residual_single = make_problem_functions(prob, sensor_points)
    sample_training_batch = make_sampler(y_train, prob)
    evaluate_y = make_evaluator(value_from_y_tx, fdm_dir, date)

    @jit
    def policy_batch(params, z_batch):
        return vmap(lambda z: find_policy_single(params, z))(z_batch)

    @jit
    def pinn_loss(params, z_batch, alpha_batch, beta_batch):
        return jnp.mean(vmap(lambda z, a, b: residual_single(params, z, a, b))(z_batch, alpha_batch, beta_batch))

    lr = float(trn["lr"])
    points_per_target = int(trn["points_per_target"])
    num_iters = int(trn["num_iters"])
    num_epochs = int(trn["num_epochs"])
    eval_every = int(trn.get("eval_every", 1))
    refresh_fraction = float(trn.get("policy_refresh_fraction", 0.1))
    early_stop_window = int(trn.get("early_stop_window", 100))
    early_stop_tol = float(trn.get("early_stop_tol", 1e-6))

    num_sensor = int(sensor_points.shape[0])
    basis_dim = int(don["basis_dim"])
    width = int(don["width"])
    depth = int(don["depth"])
    branch_layers = [num_sensor] + [width] * depth + [basis_dim]
    trunk_layers = [3] + [width] * depth + [basis_dim]

    key = random.key(seed)
    key_init, outer_key = random.split(key, 2)
    params = init_deeponet(branch_layers, trunk_layers, key_init)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state, z_batch, alpha_batch, beta_batch):
        loss, grads = value_and_grad(pinn_loss)(params, z_batch, alpha_batch, beta_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    train_error_history = []
    test_error_history = []
    loss_history = []
    t_start = time.time()
    refresh_period = max(int(num_epochs * refresh_fraction), 1)

    for it in tqdm(range(num_iters), desc=f"PI-DeepONet seed={seed}"):
        params_policy = params
        iter_key, outer_key = random.split(outer_key, 2)
        loss_log = []
        z_batch = None
        alpha_batch = None
        beta_batch = None

        for epoch in range(num_epochs):
            if epoch == 0 or (epoch + 1) % refresh_period == 0:
                iter_key, batch_key = random.split(iter_key, 2)
                z_batch = sample_training_batch(batch_key, points_per_target)
                alpha_batch, beta_batch = policy_batch(params_policy, z_batch)

            params, opt_state, loss = train_step(params, opt_state, z_batch, alpha_batch, beta_batch)
            loss_log.append(float(loss))

            if epoch >= early_stop_window:
                past_loss = loss_log[epoch - early_stop_window]
                rel_change = abs(float(loss) - past_loss) / max(abs(past_loss), 1e-8)
                if rel_change < early_stop_tol:
                    break

        loss_history.append(loss_log)
        if eval_every > 0 and (it + 1) % eval_every == 0:
            train_eval = [evaluate_y(params, y) for y in y_train]
            test_eval = [evaluate_y(params, y) for y in y_test]
            train_error_history.append([item["t0_error"] for item in train_eval])
            test_error_history.append([item["t0_error"] for item in test_eval])
            print(
                f"iter={it + 1:04d}, loss={loss_log[-1]:.4e}, "
                f"train_t0={np.mean(train_error_history[-1]):.4e}, test_t0={np.mean(test_error_history[-1]):.4e}"
            )

    final_train_eval = [evaluate_y(params, y) for y in y_train]
    final_test_eval = [evaluate_y(params, y) for y in y_test]
    train_time = time.time() - t_start
    params_num = count_params(params)
    num_points = int(y_train.shape[0] * points_per_target)

    stem = (
        f"{date}_deeponet_moving_sensor{num_sensor}_N{num_points}_iter{num_iters}_"
        f"epoch{num_epochs}_params{params_num}_basis{basis_dim}_width{width}_depth{depth}_seed{seed}"
    )
    save_path = outdir / f"{stem}.pkl"
    save_data = {
        "params": params,
        "P": {**prob, "sigma": (float(prob["sigma_scale"]) * np.eye(2)).tolist()},
        "Y_train": np.asarray(y_train),
        "Y_test": np.asarray(y_test),
        "sensor_points": np.asarray(sensor_points),
        "branch_layers": branch_layers,
        "trunk_layers": trunk_layers,
        "params_num": params_num,
        "num_points": num_points,
        "num_iters": num_iters,
        "num_epochs": num_epochs,
        "lr": lr,
        "loss_history": loss_history,
        "train_error_history": np.asarray(train_error_history),
        "test_error_history": np.asarray(test_error_history),
        "train_time": train_time,
    }
    save_pickle(save_data, save_path)

    summary = {
        "save_path": str(save_path),
        "train_time": train_time,
        "params_num": params_num,
        "Y_train": np.asarray(y_train).tolist(),
        "Y_test": np.asarray(y_test).tolist(),
        "final_train_t0_errors": {format_target_label(item["y"]): item["t0_error"] for item in final_train_eval},
        "final_test_t0_errors": {format_target_label(item["y"]): item["t0_error"] for item in final_test_eval},
        "final_train_mean_errors": {format_target_label(item["y"]): item["mean_error"] for item in final_train_eval},
        "final_test_mean_errors": {format_target_label(item["y"]): item["mean_error"] for item in final_test_eval},
    }
    with open(save_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {save_path}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the target-location DeepONet moving-obstacle baseline.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs/deeponet_run")
    parser.add_argument("--fdm_dir", type=str, default="data")
    parser.add_argument("--date", type=str, default="20260501")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed list in the config with one seed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = [int(args.seed)] if args.seed is not None else [int(s) for s in cfg["train"].get("seeds", [0])]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fdm_dir = Path(args.fdm_dir)

    for seed in seeds:
        train_one_seed(cfg, seed, outdir, fdm_dir, args.date)


if __name__ == "__main__":
    main()
