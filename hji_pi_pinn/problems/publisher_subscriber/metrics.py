from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from jax import vmap
from .problem import gen_v_single

def error_metrics(v_nn: np.ndarray, v_ref: np.ndarray, eps: float = 1e-12):
    diff = v_nn - v_ref
    l2 = np.linalg.norm(diff.ravel())
    l2_ref = np.linalg.norm(v_ref.ravel()) + eps
    rel_l2 = l2 / l2_ref
    linf = np.max(np.abs(diff))
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    return dict(rel_l2=float(rel_l2), linf=float(linf), mse=float(mse), mae=float(mae))

def eval_nn_partial_diag_slice(params, t_val: float, xg: np.ndarray, N: int, tf: float = 0.5, r: float = 1.0):
    assert (N % 2 == 1) and (N >= 3)
    v_singleN = gen_v_single(N=N, r=r, tf=tf)

    xg = np.asarray(xg)
    nx = len(xg)
    I, J = np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij")
    x0 = xg[J]
    s  = xg[I]

    XN = np.full((nx, nx, N), s[..., None], dtype=np.float64)
    XN[..., 0] = x0
    XN_flat = XN.reshape(-1, N)

    XN_j = jnp.asarray(XN_flat)
    tcol = jnp.full((XN_j.shape[0], 1), t_val)
    tx   = jnp.concatenate([tcol, XN_j], axis=1)

    v_flat = vmap(lambda z: v_singleN(z, params))(tx)
    return np.asarray(v_flat).reshape(nx, nx)

def ref_from_3d_fdm_partial_diag(V3: np.ndarray, N: int):
    assert (N % 2 == 1) and (N >= 3)
    NS = (N - 1) // 2
    nx = V3.shape[0]
    idx = np.arange(nx)
    A = V3[:, idx, idx]
    return (NS * A).T
