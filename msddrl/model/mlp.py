"""the denoiser, same structure as tabrep's MLPDiffusion:

    proj:  Linear(d_in -> dim_t)
    t:     sinusoidal(t, dim_t) -> Linear -> SiLU -> Linear
    h0  =  proj(x) + t_emb
    trunk: [Linear -> ReLU -> Dropout] for each width in trunk
    bott:  Linear(trunk[-1] -> bottleneck)   <- tap (pre-activation)
           ReLU -> Dropout
    head:  Linear(bottleneck -> d_in)

the bottleneck is the 4th and last hidden layer and its pre-activation output
is the representation. taking it before the relu means only one linear map
sits between it and the output.

dim_t defaults to trunk[0], which is 1024 in every real config (same as
tabrep) and lets the smoke model be small.

time_input=False builds the same network without the time pathway, for the
vanilla dae in exp 4: h0 = proj(x) and t is ignored.
"""

import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000):
    """tabrep's sinusoidal time encoding, unchanged.

    frequencies are spaced geometrically over max_period, output is
    [cos(...), sin(...)] concatenated.
    """
    if t.dim() == 0:
        t = t.unsqueeze(0)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Denoiser(nn.Module):
    def __init__(
        self,
        d_in,
        trunk,
        bottleneck,
        dim_t=None,
        dropout=0.0,
        max_period=10000,
        time_mlp=True,
        time_input=True,
    ):
        super().__init__()
        self.time_input = time_input
        self.d_in = d_in
        self.trunk_widths = list(trunk)
        self.bottleneck = bottleneck
        self.dim_t = dim_t if dim_t is not None else self.trunk_widths[0]
        self.max_period = max_period

        self.proj = nn.Linear(d_in, self.dim_t)
        if not time_input:
            self.time_embed = None
        elif time_mlp:
            self.time_embed = nn.Sequential(
                nn.Linear(self.dim_t, self.dim_t),
                nn.SiLU(),
                nn.Linear(self.dim_t, self.dim_t),
            )
        else:
            self.time_embed = nn.Identity()

        widths = [self.dim_t] + self.trunk_widths + [bottleneck]
        # every hidden layer is its own named Linear so the tap can hook one
        self.hidden = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(bottleneck, d_in)

    @property
    def tap_layer(self):
        """the Linear whose pre-activation output is the representation."""
        return self.hidden[-1]

    def forward(self, x, t=None, return_tap=False):
        h = self.proj(x.float())
        if self.time_input:
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, device=x.device)
            t = t.to(x.device)
            if t.dim() == 0:
                t = t.repeat(x.shape[0])
            h = h + self.time_embed(
                timestep_embedding(t, self.dim_t, self.max_period))

        tap = None
        last = len(self.hidden) - 1
        for i, layer in enumerate(self.hidden):
            h = layer(h)
            if i == last:
                tap = h
            h = self.dropout(self.activation(h))
        out = self.head(h)

        return (out, tap) if return_tap else out

    @torch.no_grad()
    def tap(self, x, t):
        """the representation for a batch (bottleneck pre-activation)."""
        return self.forward(x, t, return_tap=True)[1]

    def param_count(self):
        """number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_list(self):
        """the widths the model was built with."""
        return {
            "d_in": self.d_in,
            "proj": [self.d_in, self.dim_t],
            "hidden": [[l.in_features, l.out_features] for l in self.hidden],
            "tap": [self.tap_layer.in_features, self.tap_layer.out_features],
            "head": [self.head.in_features, self.head.out_features],
            "param_count": self.param_count(),
        }


def build(d_in, cfg, variant=None, bottleneck=None, smoke=False, trunk=None,
          time_input=True):
    """build a Denoiser from configs/model.yaml.

    an explicit `trunk` overrides the variant and the default. the queue passes
    the cell's widths so what trains is what the config says.
    """
    arch = cfg["architecture"]
    default_trunk = list(arch["trunk"])
    width = arch["bottleneck"]

    if variant is not None:
        v = arch["variants"][variant]
        default_trunk, width = list(v["trunk"]), v["bottleneck"]
    trunk = list(trunk) if trunk is not None else default_trunk
    if smoke:
        trunk = list(cfg["smoke"]["trunk"])
        width = cfg["smoke"]["bottlenecks"][0]
    if bottleneck is not None:
        width = bottleneck

    te = arch["time_embedding"]
    return Denoiser(
        d_in=d_in,
        trunk=trunk,
        bottleneck=width,
        dim_t=arch.get("dim_t"),
        dropout=arch["dropout"],
        max_period=te["max_period"],
        time_mlp=te["mlp"],
        time_input=time_input,
    )
