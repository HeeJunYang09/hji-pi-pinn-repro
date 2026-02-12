from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp
from jax import grad, hessian, vmap, random
from typing import Dict, Any

from hji_pi_pinn.core.models import forward
from hji_pi_pinn.core.data import gen_random_tx, sample_ball

@dataclass
class PSConfig:
    N: int = 5
    domain_t: tuple = (0.0, 0.5)
    domain_x: tuple = (-1.5, 1.5)
    tf: float = 0.5
    r: float = 1.0

    a: float = 1.0
    b: float = 1.0
    c: float = 0.5
    alpha: float = -2.0
    beta: float = 2.0

    base_sigma: float = 0.1
    epsilon: float = 0.3
    theta: float = float(jnp.pi/4)

def gen_v_single(N: int, r: float, tf: float):
    def v_single(tx, params):
        t = tx[0]
        x0 = tx[1]
        xi = tx[2:]
        nn = forward(params, tx)[0]
        g = 0.5 * ((N - 1) * x0**2 + jnp.sum(xi**2) - (N - 1) * r**2)
        return (tf - t) * nn + g
    return v_single

def make_sigma(cfg: PSConfig, seed: int = 0):
    key = random.PRNGKey(cfg.N * 2 * (seed + 1))
    rand_up = random.uniform(key, (cfg.N, cfg.N), minval=0.0, maxval=cfg.epsilon)
    upper = jnp.triu(rand_up, k=1)
    noise = upper + upper.T
    noise = noise * (1 - jnp.eye(cfg.N))
    return cfg.base_sigma * jnp.eye(cfg.N) + noise

def build_ps_components(cfg: PSConfig, seed: int = 0) -> Dict[str, Any]:
    assert cfg.N % 2 == 1 and cfg.N >= 3
    NS = (cfg.N - 1) // 2

    sigma = make_sigma(cfg, seed=seed)
    R = jnp.array([[jnp.cos(cfg.theta), -jnp.sin(cfg.theta)],
                   [jnp.sin(cfg.theta),  jnp.cos(cfg.theta)]], dtype=jnp.float32)

    v_single = gen_v_single(cfg.N, cfg.r, cfg.tf)

    def v_spatial_only(x, t_fixed, params):
        t_fixed = jnp.atleast_1d(t_fixed)
        return v_single(jnp.concatenate([t_fixed, x]), params)

    def hess_x(x, t, params):
        return hessian(lambda xx: v_spatial_only(xx, t, params))(x)

    def loss_pi(params, batch, policy):
        grid_tx = batch["grid_tx"]
        u = policy["u"]
        d = policy["d"]

        def residual(tx, u_i, d_i):
            t = tx[:1]
            x = tx[1:]
            g = grad(v_single)(tx, params)
            dv_dt = g[0]
            grad_x = g[1:]
            p_sub = grad_x[1:]
            H = hess_x(x, t, params)

            e1 = jnp.eye(cfg.N, 1, dtype=jnp.float32)
            oneN = jnp.ones((cfg.N, 1), dtype=jnp.float32)
            xvec = x.reshape(cfg.N, 1)

            lin = (e1 @ e1.T - oneN @ e1.T + cfg.a * jnp.eye(cfg.N, dtype=jnp.float32)) @ xvec
            non1 = cfg.alpha * jnp.sin(x[0]) * e1
            non2 = -cfg.beta * x[0] * (1.0 - e1)
            non = (non1 + non2) * (xvec**2)
            f = (lin + non).reshape(-1)

            Bu = cfg.b * jnp.sum(u_i * p_sub)
            p_blocks = p_sub.reshape(NS, 2)
            d_blocks = d_i.reshape(NS, 2)
            q_blocks = p_blocks @ R
            Cd = cfg.c * jnp.sum(d_blocks * q_blocks)

            diff = 0.5 * jnp.trace(sigma @ sigma.T @ H)
            return (dv_dt + jnp.dot(grad_x, f) + Bu + Cd + diff) ** 2

        return jnp.mean(vmap(residual)(grid_tx, u, d))

    def policy_update(params, batch):
        grid_tx = batch["grid_tx"]

        def sign_pm1(z):
            return jnp.where(z >= 0, 1.0, -1.0)

        def one(tx):
            g = grad(v_single)(tx, params)
            p_sub = g[2:]                    # (N-1,)
            u = -sign_pm1(p_sub)             # min over ||u||_inf<=1

            p_blocks = p_sub.reshape(NS, 2)
            q_blocks = p_blocks @ R
            d = sign_pm1(q_blocks).reshape(-1)   # max over ||d||_inf<=1
            return u, d

        u, d = vmap(one)(grid_tx)
        return {"u": u, "d": d}

    def loss_direct(params, batch, _policy_unused=None):
        grid_tx = batch["grid_tx"]

        def residual(tx):
            t = tx[:1]
            x = tx[1:]
            g = grad(v_single)(tx, params)
            dv_dt = g[0]
            grad_x = g[1:]
            dv_dxi = grad_x[1:]

            H = hess_x(x, t, params)

            e1 = jnp.eye(cfg.N, 1, dtype=jnp.float32)
            oneN = jnp.ones((cfg.N, 1), dtype=jnp.float32)
            xvec = x.reshape(cfg.N, 1)

            lin = (e1 @ e1.T - oneN @ e1.T + cfg.a * jnp.eye(cfg.N, dtype=jnp.float32)) @ xvec
            non1 = cfg.alpha * jnp.sin(x[0]) * e1
            non2 = -cfg.beta * x[0] * (1.0 - e1)
            non = (non1 + non2) * (xvec**2)
            f = (lin + non).reshape(-1)

            term_u = -cfg.b * jnp.sum(jnp.abs(dv_dxi))
            p_blocks = dv_dxi.reshape(NS, 2)
            q_blocks = p_blocks @ R
            term_d = cfg.c * jnp.sum(jnp.abs(q_blocks))

            diff = 0.5 * jnp.trace(sigma @ sigma.T @ H)
            return (dv_dt + jnp.dot(grad_x, f) + term_u + term_d + diff) ** 2

        return jnp.mean(vmap(residual)(grid_tx))

    def sample_batch(key, Ni):
        grid_tx = gen_random_tx(key, Ni, cfg.N, cfg.domain_x, cfg.domain_t)
        return {"grid_tx": grid_tx}

    def init_policy(key, batch):
        Ni = batch["grid_tx"].shape[0]
        k1, k2 = random.split(key, 2)
        u_keys = random.split(k1, Ni)
        d_keys = random.split(k2, Ni)
        u = jnp.stack([sample_ball(k, 1.0, cfg.N - 1) for k in u_keys], axis=0)
        d = jnp.stack([sample_ball(k, 1.0, cfg.N - 1) for k in d_keys], axis=0)
        return {"u": u, "d": d}

    return dict(
        sigma=sigma,
        v_single=v_single,
        loss_pi=loss_pi,
        loss_direct=loss_direct,
        policy_update=policy_update,
        sample_batch=sample_batch,
        init_policy=init_policy,
    )
