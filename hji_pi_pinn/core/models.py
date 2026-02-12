from __future__ import annotations
import jax.numpy as jnp
from jax import jit, random
from jax.nn import initializers

@jit
def forward(params, data):
    x = data
    for (w, b) in params[:-1]:
        x = jnp.sin(jnp.matmul(x, w) + b)
    w, b = params[-1]
    return jnp.matmul(x, w) + b

def init_params(layers_dims, key):
    initializer = initializers.glorot_uniform()
    keys = random.split(key, len(layers_dims) - 1)

    def _init(in_dim, out_dim, k):
        w = initializer(k, (in_dim, out_dim), jnp.float32)
        b = jnp.zeros((out_dim,), dtype=jnp.float32)
        return (w, b)

    return [_init(i, o, k) for i, o, k in zip(layers_dims[:-1], layers_dims[1:], keys)]

def count_params(params) -> int:
    return int(sum(w.size + b.size for w, b in params))
