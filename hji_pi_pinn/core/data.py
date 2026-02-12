from __future__ import annotations
import jax.numpy as jnp
from jax import random

def gen_random_tx(key, num, dim_x, domain_x, domain_t, dtype=jnp.float32):
    keys = random.split(key, dim_x + 1)
    t = random.uniform(keys[0], (num,), minval=domain_t[0], maxval=domain_t[1], dtype=dtype)
    xs = [random.uniform(keys[i+1], (num,), minval=domain_x[0], maxval=domain_x[1], dtype=dtype) for i in range(dim_x)]
    return jnp.stack([t, *xs], axis=-1)

def sample_ball(key, radius: float, dim: int):
    k1, k2 = random.split(key, 2)
    v = random.normal(k1, (dim,))
    v = v / (jnp.linalg.norm(v) + 1e-12)
    r = random.uniform(k2, ()) ** (1.0 / dim)
    return radius * r * v
