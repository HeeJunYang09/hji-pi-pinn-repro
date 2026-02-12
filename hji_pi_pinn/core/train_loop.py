from __future__ import annotations
import time
from typing import Callable, Dict, Any

from jax import random, value_and_grad, jit
import optax

from .models import init_params

def train_pi_pinn(
    *,
    key,
    layers,
    loss_fn: Callable,
    policy_update_fn: Callable,
    sample_batch_fn: Callable,
    num_iters: int,
    num_epochs: int,
    lr: float,
    init_policy_fn: Callable,
) -> Dict[str, Any]:
    """Generic PI-PINN driver (paper-repro version)."""
    k_p, key = random.split(key, 2)
    params = init_params(layers, k_p)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    key, kb = random.split(key)
    batch = sample_batch_fn(kb)

    key, kp = random.split(key)
    policy = init_policy_fn(kp, batch)

    @jit
    def step(params, opt_state, batch, policy):
        loss, grads = value_and_grad(loss_fn)(params, batch, policy)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    start = time.time()
    last_loss = None
    for it in range(num_iters):
        for ep in range(num_epochs):
            params, opt_state, last_loss = step(params, opt_state, batch, policy)

        key, kb = random.split(key)
        batch = sample_batch_fn(kb)
        policy = policy_update_fn(params, batch)

    return {
        "params": params,
        "train_time": time.time() - start,
        "last_loss": float(last_loss) if last_loss is not None else None,
    }
