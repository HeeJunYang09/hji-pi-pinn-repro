from __future__ import annotations
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse, yaml
from pathlib import Path
import numpy as np
from jax import random, value_and_grad, jit
import optax
from tqdm import tqdm
import time

from hji_pi_pinn.core.models import init_params, count_params
from hji_pi_pinn.core.io import save_pickle
from hji_pi_pinn.problems.publisher_subscriber import PSConfig, build_ps_components
from hji_pi_pinn.problems.publisher_subscriber import error_metrics, eval_nn_partial_diag_slice, ref_from_3d_fdm_partial_diag


def compute_fdm_metrics(params, cfg: PSConfig, v_fdm, t_idx: int = 0):
    V3 = v_fdm[t_idx]  # (nx,nx,nx)
    nx = V3.shape[0]
    xg = np.linspace(-0.5, 0.5, nx)
    v_nn = eval_nn_partial_diag_slice(params, 0.0, xg, cfg.N, tf=cfg.tf, r=cfg.r)
    v_ref = ref_from_3d_fdm_partial_diag(V3, cfg.N)
    return error_metrics(v_nn, v_ref)

def make_pbar(total_steps, desc="train"):
    pbar = tqdm(total=100, desc=desc, ncols=90)
    last_bucket = -1
    def update(done_steps, loss=None):
        nonlocal last_bucket
        pct = int(100 * done_steps / max(total_steps, 1))
        bucket = (pct // 5) * 5
        if bucket != last_bucket:
            last_bucket = bucket
            pbar.n = bucket
            if loss is not None:
                pbar.set_postfix(loss=float(loss))
            pbar.refresh()
    return pbar, update


def build_ps_info(
    *,
    mode: str,
    points: int,
    num_iters: int,
    num_epochs: int,
    num_params: int,
    n_layers: int,
    width: int,
    prob: dict,
    n_dim: int,
):
    # Match existing PS output schemas:
    # - PI: points, iter, epoch, ...
    # - Direct: points, epoch, ...
    info = dict(points=points)
    if mode == "pi":
        info["iter"] = num_iters
        info["epoch"] = num_epochs
    else:
        info["epoch"] = num_iters * num_epochs

    info.update(
        dict(
            num_params=num_params,
            layer=n_layers,
            width=width,
            a=prob.get("a", None),
            b=prob.get("b", None),
            c=prob.get("c", None),
            alpha=prob.get("alpha", None),
            beta=prob.get("beta", None),
            dimension=f"{n_dim}D",
            epsilon=prob.get("epsilon", 0),
            mode=mode,
        )
    )
    return info


def format_epsilon_token(epsilon) -> str:
    # Filename token convention:
    # 0 -> "0", 0.1 -> "01", 0.3 -> "03", 0.5 -> "05"
    s = f"{float(epsilon):g}"
    if s == "0":
        return "0"
    return s.replace(".", "").replace("-", "m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs/ps_run")
    p.add_argument("--outname", type=str, default="auto")
    p.add_argument("--mode", type=str, choices=["pi", "direct"], default=None)
    p.add_argument("--fdm_path", type=str, default=None)
    p.add_argument("--eval_every_outer", type=int, default=1)
    args = p.parse_args()

    cfg_all = yaml.safe_load(open(args.config, "r"))
    prob = cfg_all.get("problem", {})
    trn  = cfg_all.get("train", {})

    cfg = PSConfig(**prob)
    mode = args.mode or trn.get("mode", "pi")
    if mode not in ("pi", "direct"):
        raise ValueError(f"Invalid mode='{mode}'. Expected 'pi' or 'direct'.")
    seed = int(trn.get("seed", 0))

    comps = build_ps_components(cfg, seed=seed)

    unit = int(trn.get("unit", 64))
    layers = [cfg.N + 1, unit, unit, unit, 1]

    Ni = int(trn.get("Ni", 25000))
    num_iters = int(trn.get("num_iters", 500))
    num_epochs = int(trn.get("num_epochs", 5000))
    lr = float(trn.get("lr", 1e-3))

    v_fdm = None
    if args.fdm_path:
        v_fdm = np.load(args.fdm_path)
        # accept shape (5,nx,nx,nx) or (nx,nx,nx) (single snapshot)
        if v_fdm.ndim == 3:
            v_fdm = v_fdm[None, ...]

    key = random.PRNGKey(seed)
    k_p, k_b, k_pol = random.split(key, 3)
    params = init_params(layers, k_p)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    batch = comps["sample_batch"](k_b, Ni)
    policy = comps["init_policy"](k_pol, batch) if mode == "pi" else None

    t0 = time.time()
    l2_history = []
    if v_fdm is not None:
        m = compute_fdm_metrics(params, cfg, v_fdm, t_idx=0)
        l2_history.append(m["rel_l2"])

    if mode == "pi":
        loss_fn = comps["loss_pi"]
        policy_update = comps["policy_update"]

        @jit
        def step(params, opt_state, batch, policy):
            loss, grads = value_and_grad(loss_fn)(params, batch, policy)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        total_steps = num_iters * num_epochs
        pbar, upd = make_pbar(total_steps, desc=f"PS-{mode}")
        done = 0
        for it in range(num_iters):
            for _ in range(num_epochs):
                params, opt_state, loss = step(params, opt_state, batch, policy)
                done += 1
                upd(done, loss)

            # resample + policy update (outer iteration)
            k_b, k_pol = random.split(k_b, 2)
            batch = comps["sample_batch"](k_b, Ni)
            policy = policy_update(params, batch)

            if v_fdm is not None and (it + 1) % args.eval_every_outer == 0:
                m = compute_fdm_metrics(params, cfg, v_fdm, t_idx=0)
                l2_history.append(m["rel_l2"])
        pbar.close()
    else:
        loss_fn = comps["loss_direct"]
        total_epochs = num_iters * num_epochs
        eval_block = max(1, total_epochs // num_iters)  # equals num_epochs

        @jit
        def step(params, opt_state, batch):
            loss, grads = value_and_grad(loss_fn)(params, batch, None)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        total_steps = total_epochs
        pbar, upd = make_pbar(total_steps, desc=f"PS-{mode}")
        done = 0
        for ep in range(total_epochs):
            if (ep + 1) % 500 == 0:
                k_b, _ = random.split(k_b, 2)
                batch = comps["sample_batch"](k_b, Ni)

            params, opt_state, loss = step(params, opt_state, batch)
            done += 1
            upd(done, loss)
            if v_fdm is not None and (ep + 1) % eval_block == 0:
                block_id = (ep + 1) // eval_block
                if block_id % args.eval_every_outer == 0:
                    m = compute_fdm_metrics(params, cfg, v_fdm, t_idx=0)
                    l2_history.append(m["rel_l2"])
        pbar.close()
    train_time = time.time() - t0
    info = build_ps_info(
        mode=mode,
        points=Ni,
        num_iters=num_iters,
        num_epochs=num_epochs,
        num_params=count_params(params),
        n_layers=len(layers) - 1,
        width=unit,
        prob=prob,
        n_dim=cfg.N,
    )

    out = {
        "params": params,
        "l2_history": l2_history,
        "train_time": train_time,
        "info": info,
    }

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    outname = args.outname
    if outname == "auto":
        eps_token = format_epsilon_token(prob.get("epsilon", 0))
        if mode == "pi":
            outname = f"PI_N{Ni}_it{num_iters}_ep{num_epochs}_epsilon{eps_token}_seed{seed}.pkl"
        else:
            outname = f"Direct_N{Ni}_ep{num_iters * num_epochs}_epsilon{eps_token}_seed{seed}.pkl"
    save_path = outdir / outname
    save_pickle(out, save_path)
    print(f"[saved] {save_path}")

if __name__ == "__main__":
    main()
