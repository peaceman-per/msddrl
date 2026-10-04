"""representation extraction.

the noise used at extraction is one fixed matrix per split (a seeded draw
the size of the split), reused at every grid point. so two grid points differ
in (t_true, t_label) and not in which eps they got.

    z = t_true * X + (1 - t_true) * E_split

the network is told t_label, which doesnt have to equal t_true. the
representation is the bottleneck's pre-activation output.

the matrix is not saved to disk, the seeded generator rebuilds it.
"""

import hashlib

import torch

# split index train=0, val=1, test=2, so the seeds are 128, 129, 130
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}
NOISE_SEED_BASE = 128


def split_seed(split):
    if split not in SPLIT_INDEX:
        raise ValueError(f"unknown split {split!r}")
    return NOISE_SEED_BASE + SPLIT_INDEX[split]


def noise_matrix(split, n_rows, d_in, device="cpu"):
    """E_split, rebuilt on demand. the same for any two cells on a split."""
    g = torch.Generator(device="cpu").manual_seed(split_seed(split))
    e = torch.randn(n_rows, d_in, generator=g, dtype=torch.float32)
    return e.to(device)


def noise_hash(e):
    """a digest of the matrix, to check two cells share a draw."""
    return hashlib.sha256(e.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


@torch.no_grad()
def extract(model, x, idx, t_true, t_label, e_split, batch_size=8192, device=None):
    """corrupt at t_true, label at t_label, return (tap output, idx).

    runs on cpu by default so a probe worker doesnt need a gpu.
    """
    device = device or next(model.parameters()).device
    if len(x) != len(e_split):
        raise AssertionError(
            f"noise matrix has {len(e_split)} rows, split has {len(x)}"
        )
    if len(x) != len(idx):
        raise AssertionError("row index does not match the feature matrix")

    was_training = model.training
    model.eval()

    out = []
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        xb = x[start:stop].to(device)
        eb = e_split[start:stop].to(device)
        # written out so t_true = 1 gives z == x exactly
        z = t_true * xb + (1.0 - t_true) * eb
        t = torch.full((len(xb),), float(t_label), device=device)
        out.append(model(z, t, return_tap=True)[1].detach().cpu())

    if was_training:
        model.train()
    return torch.cat(out, dim=0), idx
