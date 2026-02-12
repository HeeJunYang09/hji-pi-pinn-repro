from __future__ import annotations

import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse, yaml
from pathlib import Path
from jax import random

from hji_pi_pinn.core.train_loop import train_pi_pinn
from hji_pi_pinn.core.io import save_pickle
from hji_pi_pinn.core.models import count_params
from hji_pi_pinn.problems.path_planning import PathPlanningConfig, build_path_planning_components


def build_path_info(
    *,
    points: int,
    num_iters: int,
    num_epochs: int,
    num_params: int,
    n_layers: int,
    width: int,
    cfg: PathPlanningConfig,
):
    # Keep key order/type consistent with existing path_run reference files.
    return dict(
        points=int(points),
        iter=int(num_iters),
        epoch=int(num_epochs),
        num_params=int(num_params),
        layer=int(n_layers),
        width=int(width),
        lambda_1=float(cfg.lambda_1),
        lambda_2=float(cfg.lambda_2),
        lambda_3=float(cfg.lambda_3),
        delta=float(cfg.delta),
        epsilon=float(cfg.epsilon),
        dimension=f"{int(cfg.dim_x)}D",
        sigma_scale=float(cfg.sigma_scale),
        sigma_dim=int(cfg.dim_x),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs/path_run")
    p.add_argument("--outname", type=str, default="auto")
    args = p.parse_args()

    cfg_all = yaml.safe_load(open(args.config, "r"))
    prob = cfg_all.get("problem", {})
    trn  = cfg_all.get("train", {})

    cfg = PathPlanningConfig(**prob)
    comps = build_path_planning_components(cfg)

    unit = int(trn.get("unit", 64))
    layers = [cfg.dim_x + 1, unit, unit, unit, 1]

    Ni = int(trn.get("Ni", 20000))
    num_iters = int(trn.get("num_iters", 200))
    num_epochs = int(trn.get("num_epochs", 5000))
    lr = float(trn.get("lr", 1e-3))
    seed = int(trn.get("seed", 0))

    key = random.PRNGKey(seed)
    train_out = train_pi_pinn(
        key=key,
        layers=layers,
        loss_fn=comps["loss_fn"],
        policy_update_fn=comps["policy_update_fn"],
        sample_batch_fn=lambda k: comps["sample_batch_fn"](k, Ni),
        num_iters=num_iters,
        num_epochs=num_epochs,
        lr=lr,
        init_policy_fn=lambda k, b: comps["init_policy_fn"](k, b),
    )

    info = build_path_info(
        points=Ni,
        num_iters=num_iters,
        num_epochs=num_epochs,
        num_params=count_params(train_out["params"]),
        n_layers=len(layers) - 1,
        width=unit,
        cfg=cfg,
    )
    out = {"params": train_out["params"], "info": info}

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    outname = args.outname
    if outname == "auto":
        outname = f"PI_N{Ni}_it{num_iters}_ep{num_epochs}_seed{seed}.pkl"
    save_path = outdir / outname
    save_pickle(out, save_path)
    print(f"[saved] {save_path}")

if __name__ == "__main__":
    main()
