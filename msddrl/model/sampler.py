"""probability-flow ode sampler, only used by exp 1.

since z_t = t*x + (1-t)*eps, the derivative of the path is the velocity,
dz/dt = x - eps = v. so generation integrates

    dz/dt = v_theta(z, t)

from t = delta (noise) to 1-delta (data) with euler steps, starting at
z(delta) = eps ~ N(0, I).

whatever the network predicts is converted to v at every step, so all three
pred modes use the same loop. the sample is the euler state at t = 1-delta.
"""

import torch

from msddrl.model.conversions import to_velocity


@torch.no_grad()
def sample(
    model,
    n_samples,
    d_in,
    pred_mode,
    steps,
    delta,
    batch_size=None,
    device=None,
    generator=None,
):
    """generate n_samples rows in the encoding space."""
    device = device or next(model.parameters()).device
    batch_size = batch_size or n_samples
    was_training = model.training
    model.eval()

    out = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        out.append(
            _sample_batch(model, b, d_in, pred_mode, steps, delta, device, generator)
        )
        remaining -= b

    if was_training:
        model.train()
    return torch.cat(out, dim=0)


def _sample_batch(model, b, d_in, pred_mode, steps, delta, device, generator):
    z = torch.randn(b, d_in, device=device, generator=generator)

    # `steps` uniform euler steps over [delta, 1-delta]
    ts = torch.linspace(delta, 1.0 - delta, steps + 1, device=device)
    for i in range(steps):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        out = model(z, t.expand(b))
        v = to_velocity(out, z, t.expand(b), pred_mode)
        z = z + v * dt

    if not torch.isfinite(z).all():
        raise FloatingPointError("sampler produced non-finite values")
    return z
