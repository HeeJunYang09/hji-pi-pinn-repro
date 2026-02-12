from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp
from jax import grad, hessian, vmap, random

from hji_pi_pinn.core.models import forward
from hji_pi_pinn.core.data import gen_random_tx, sample_ball

@dataclass
class PathPlanningConfig:
    domain_t: tuple = (0.0, 1.0)
    domain_x: tuple = (-1.0, 1.0)
    dim_x: int = 2
    tf: float = 1.0

    lambda_1: float = 0.1
    lambda_2: float = 100.0
    lambda_3: float = 10.0
    delta: float = 0.1
    epsilon: float = 0.3

    x_goal: tuple = (0.9, 0.9)
    sigma_scale: float = 0.1

def build_path_planning_components(cfg: PathPlanningConfig):
    x_goal = jnp.array(cfg.x_goal, dtype=jnp.float32)
    sigma = cfg.sigma_scale * jnp.eye(cfg.dim_x, dtype=jnp.float32)

    def phi_fn(t_vec, x):
        t = t_vec[0]
        x_obs = jnp.array([0.5 * jnp.cos(jnp.pi * t), 0.5 * jnp.sin(jnp.pi * t)], dtype=jnp.float32)
        return jnp.exp(-jnp.linalg.norm(x - x_obs)**2 / (2.0 * cfg.epsilon**2))

    def v_single(tx, params):
        t = tx[0]
        x = tx[1:]
        nn = forward(params, tx)[0]
        g = cfg.lambda_3 * jnp.sum((x - x_goal) ** 2)
        return (cfg.tf - t) * nn + g

    def v_spatial_only(x, t_fixed, params):
        t_fixed = jnp.atleast_1d(t_fixed)
        return v_single(jnp.concatenate([t_fixed, x]), params)

    def hess_x(x, t, params):
        return hessian(lambda xx: v_spatial_only(xx, t, params))(x)

    def policy_update_fn(params, batch):
        grid_tx = batch["grid_tx"]
        def one(tx):
            g = grad(v_single)(tx, params)
            p = g[1:]
            pn = jnp.linalg.norm(p) + 1e-10
            alpha_un = -p / (2.0 * cfg.lambda_1)
            alpha = jnp.where(pn <= 2.0 * cfg.lambda_1, alpha_un, -p / pn)
            beta = cfg.delta * p / pn
            return alpha, beta
        a, b = vmap(one)(grid_tx)
        return {"alpha": a, "beta": b}

    def loss_fn(params, batch, policy):
        grid_tx = batch["grid_tx"]
        alpha = policy["alpha"]
        beta  = policy["beta"]
        def residual(tx, a, b):
            t = tx[:1]
            x = tx[1:]
            g = grad(v_single)(tx, params)
            dv_dt = g[0]
            dv_dx = g[1:]
            H = hess_x(x, t, params)
            phi = phi_fn(t, x)
            diff = 0.5 * jnp.trace(sigma @ sigma.T @ H)
            return (dv_dt + cfg.lambda_1*jnp.sum(a*a) + cfg.lambda_2*phi + jnp.dot(dv_dx, a+b) + diff) ** 2
        return jnp.mean(vmap(residual)(grid_tx, alpha, beta))

    def sample_batch_fn(key, Ni):
        grid_tx = gen_random_tx(key, Ni, cfg.dim_x, cfg.domain_x, cfg.domain_t)
        return {"grid_tx": grid_tx}

    def init_policy_fn(key, batch):
        Ni = batch["grid_tx"].shape[0]
        k1, k2 = random.split(key, 2)
        keys1 = random.split(k1, Ni)
        keys2 = random.split(k2, Ni)
        alpha = jnp.stack([sample_ball(k, 1.0, cfg.dim_x) for k in keys1], axis=0)
        beta  = jnp.stack([sample_ball(k, cfg.delta, cfg.dim_x) for k in keys2], axis=0)
        return {"alpha": alpha, "beta": beta}

    return dict(
        sigma=sigma,
        v_single=v_single,
        loss_fn=loss_fn,
        policy_update_fn=policy_update_fn,
        sample_batch_fn=sample_batch_fn,
        init_policy_fn=init_policy_fn,
    )
