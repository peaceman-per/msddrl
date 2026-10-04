"""the linear interpolant.

    z_t = t*x + (1-t)*eps,   eps ~ N(0, I)

t = 1 is clean data, t = 0 is pure noise. the equivalent noise level sigma
comes from factoring out t:

    z_t = t * (x + ((1-t)/t) * eps)   =>   sigma(t) = (1-t)/t,  t(sigma) = 1/(1+sigma)
"""

from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def interpolate(x, eps, t):
    """z_t = t*x + (1-t)*eps.

    written out literally so t = 1 returns x exactly (1.0*x + 0.0*eps).
    """
    t = _broadcast(t, x)
    return t * x + (1.0 - t) * eps


def velocity(x, eps):
    """v = x - eps, which is dz/dt."""
    return x - eps


def _broadcast(t, x):
    """shape a per-row t (or a scalar) to broadcast against x."""
    if not torch.is_tensor(t):
        return torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if t.dim() == 0:
        return t.to(dtype=x.dtype, device=x.device)
    return t.reshape(-1, *([1] * (x.dim() - 1))).to(dtype=x.dtype, device=x.device)


def sample_t(n, delta, generator=None, device=None, dtype=torch.float32):
    """t ~ U(delta, 1-delta), used for training in every cell.

    uniform on purpose: a non-uniform t would weight the loss differently in
    each of the three loss spaces and confound the comparison.
    """
    u = torch.rand(n, generator=generator, device=device, dtype=dtype)
    return delta + (1.0 - 2.0 * delta) * u


def sigma_to_t(sigma):
    return 1.0 / (1.0 + sigma)


def t_to_sigma(t):
    return (1.0 - t) / t


def grid_values(cfg=None):
    """the 12 t values of the (t_true, t_label) grid, descending from 1.0.

    sigma = 0 (t = 1, clean) plus eleven geometric points
    sigma_k = sigma_base * 2^(k/2) for k = 0..10.
    """
    cfg = cfg or load_config()
    g = cfg["interpolant"]["grid"]
    values = [1.0]
    for k in range(g["n_geometric"]):
        values.append(sigma_to_t(g["sigma_base"] * 2.0 ** (k / 2.0)))
    return values


def grid_pairs(cfg=None):
    """all (t_true, t_label) pairs, 144 of them."""
    values = grid_values(cfg)
    return [(a, b) for a in values for b in values]
