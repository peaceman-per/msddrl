"""conversions between the x, eps and v spaces.

the network outputs one tensor. pred_mode says how to read it and loss_space
says where the mse is taken. all nine combinations go through `loss()`.

with z_t = t*x + (1-t)*eps and v = x - eps:

    from x_hat:    eps_hat = (z_t - t*x_hat) / (1-t)
                   v_hat   = x_hat - eps_hat = (x_hat - z_t) / (1-t)
    from eps_hat:  x_hat   = (z_t - (1-t)*eps_hat) / t
                   v_hat   = x_hat - eps_hat
    from v_hat:    x_hat   = z_t + (1-t)*v_hat
                   eps_hat = z_t - t*v_hat

at a fixed t the three residuals are proportional:

    ||eps_hat - eps||^2 = t^2/(1-t)^2 * ||x_hat - x||^2
    ||v_hat   - v  ||^2 =   1/(1-t)^2 * ||x_hat - x||^2

so the loss curves only differ by a scale factor between spaces.
"""

import torch

from msddrl.model.interpolant import _broadcast

SPACES = ("x", "eps", "v")


def _check(space, name):
    if space not in SPACES:
        raise ValueError(f"{name} must be one of {SPACES}, got {space!r}")


def to_all_spaces(out, z_t, t, pred_mode):
    """read the network output as pred_mode, return (x_hat, eps_hat, v_hat)."""
    _check(pred_mode, "pred_mode")
    t = _broadcast(t, z_t)
    one_minus_t = 1.0 - t

    if pred_mode == "x":
        x_hat = out
        eps_hat = (z_t - t * x_hat) / one_minus_t
        v_hat = x_hat - eps_hat
    elif pred_mode == "eps":
        eps_hat = out
        x_hat = (z_t - one_minus_t * eps_hat) / t
        v_hat = x_hat - eps_hat
    else:
        v_hat = out
        x_hat = z_t + one_minus_t * v_hat
        eps_hat = z_t - t * v_hat

    return x_hat, eps_hat, v_hat


def to_velocity(out, z_t, t, pred_mode):
    """just v_hat, which is all the sampler needs."""
    return to_all_spaces(out, z_t, t, pred_mode)[2]


def targets(x, eps):
    """the ground truth in each space, same keys as to_all_spaces."""
    return {"x": x, "eps": eps, "v": x - eps}


def loss_per_sample(pred_mode, loss_space, out, z_t, t, x, eps):
    """squared error in loss_space, averaged over features, per row.

    per row because the conversion scales the residual by t/(1-t) or 1/(1-t),
    so when loss_space != pred_mode a batch's losses are very right-tailed and
    only a per-row max/median shows it.
    """
    _check(pred_mode, "pred_mode")
    _check(loss_space, "loss_space")

    x_hat, eps_hat, v_hat = to_all_spaces(out, z_t, t, pred_mode)
    prediction = {"x": x_hat, "eps": eps_hat, "v": v_hat}[loss_space]
    target = targets(x, eps)[loss_space]
    squared = (prediction - target) ** 2
    return squared.reshape(squared.shape[0], -1).mean(dim=1)


def loss(pred_mode, loss_space, out, z_t, t, x, eps):
    """mse in loss_space after converting from pred_mode."""
    return loss_per_sample(pred_mode, loss_space, out, z_t, t, x, eps).mean()


def residual_scale(t, loss_space, device=None, dtype=torch.float32):
    """factor between a residual in loss_space and the same one in x space.

    ||r_space||^2 = residual_scale(t, space)^2 * ||r_x||^2
    """
    t = torch.as_tensor(t, device=device, dtype=dtype)
    if loss_space == "x":
        return torch.ones_like(t)
    if loss_space == "eps":
        return t / (1.0 - t)
    return 1.0 / (1.0 - t)
