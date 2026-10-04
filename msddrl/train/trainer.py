"""the training loop.

one cell = one training run. the budget is fixed in epochs (not iterations)
so every dataset gets the same number of passes over its rows. no early
stopping: losses in x, eps and v space are not comparable, so stopping on the
loss would give the nine combinations different budgets.

the exception is the vanilla dae in exp 4 (method="dae"). it has one loss in
one space so it early stops on validation loss. it corrupts with
z = x + sigma * eps, reconstructs x with mse, has no time input and keeps
its best checkpoint in ckpt_best.pt.

randomness is seeded per epoch from (seed, epoch), so resuming at epoch k
gives the same result as an uninterrupted run.
"""

import csv
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from msddrl.data.loader import load_fold, load_split
from msddrl.model import conversions, interpolant
from msddrl.model.mlp import build as build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"
MODEL_CONFIG = REPO_ROOT / "configs" / "model.yaml"


def load_configs(train_path=TRAIN_CONFIG, model_path=MODEL_CONFIG):
    with open(train_path) as f:
        train = yaml.safe_load(f)
    with open(model_path) as f:
        model = yaml.safe_load(f)
    return train, model


@dataclass
class TrainConfig:
    """everything a training run needs."""

    dataset: str
    pred_mode: str = "v"
    loss_space: str = "v"
    bottleneck: int = 512
    arch_variant: str = None
    trunk: list = None
    seed: int = 1
    epochs: int = 15000
    batch_size: int = 4096
    eval_every: int = 150
    lr: float = 1e-4
    weight_decay: float = 5e-4
    delta: float = 1e-3
    target_in_input: bool = False
    subsample_train: int = None
    smoke: bool = False
    val_t_grid: list = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    eps_seed_base: int = 1000
    instrument: bool = True
    # fp32 or bf16. bf16 is autocast around the training forward pass only,
    # the loss, optimizer, weights and validation all stay fp32
    precision: str = "fp32"
    # rotating fold (exp 4 regression): train on the fold's train part and
    # validate on the fixed val rows under the fold's transforms
    fold: int = None
    # "ddae", or "dae" for the vanilla dae in exp 4 (gaussian noise at
    # dae_sigma, x reconstruction, no time input, early stopping after
    # `patience` validation checks without improvement)
    method: str = "ddae"
    dae_sigma: float = None
    patience: int = None

    @classmethod
    def from_configs(cls, dataset, train_cfg, model_cfg, smoke=False, **overrides):
        b, o, v = train_cfg["budget"], train_cfg["optimizer"], train_cfg["validation"]
        kwargs = dict(
            dataset=dataset,
            epochs=b["epochs"],
            batch_size=b["batch_size"],
            eval_every=b["eval_every"],
            lr=o["lr"],
            weight_decay=o["weight_decay"],
            delta=model_cfg["interpolant"]["delta"],
            bottleneck=model_cfg["architecture"]["bottleneck"],
            val_t_grid=list(v["t_grid"]),
            eps_seed_base=v["eps_seed_base"],
            instrument=train_cfg["instrumentation"]["enabled"],
            smoke=smoke,
        )
        if smoke:
            s = train_cfg["smoke"]
            kwargs.update(epochs=s["epochs"], eval_every=s["eval_every"])
        kwargs.update(overrides)
        return cls(**kwargs)


def _epoch_generator(seed, epoch, device, tag=0):
    """generator seeded from (seed, epoch), so resume is exact."""
    g = torch.Generator(device=device)
    g.manual_seed((seed * 1_000_003 + epoch * 10_007 + tag) % (2**63 - 1))
    return g


class Trainer:
    def __init__(self, cfg: TrainConfig, out_dir, device=None, heartbeat=None):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # no-op unless the queue passes a real heartbeat
        self.heartbeat = heartbeat or (lambda: None)

        if cfg.method not in ("ddae", "dae"):
            raise ValueError(f"unknown method {cfg.method!r}")
        if cfg.method == "dae" and (cfg.dae_sigma is None or cfg.patience is None):
            raise ValueError("a DAE needs dae_sigma and patience")

        self._seed_globals(cfg.seed)
        self._load_data()
        self._build()
        self.epoch = 0
        # early stopping state (dae only), saved in the checkpoint
        self.best_val = float("inf")
        self.best_epoch = 0
        self.bad_checks = 0

    # -- setup -------------------------------------------------------------

    def _seed_globals(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_data(self):
        c = self.cfg
        if c.fold is None:
            x_train, _, _, info = load_split(c.dataset, "train", c.target_in_input)
            x_val, _, _, _ = load_split(c.dataset, "val", c.target_in_input)
        else:
            x_train, _, _, info = load_fold(c.dataset, c.fold, "train", c.target_in_input)
            x_val, _, _, _ = load_fold(c.dataset, c.fold, "val", c.target_in_input)

        if c.smoke:
            train_rows = self._smoke_rows("train_rows")
            val_rows = self._smoke_rows("val_rows")
            x_train, x_val = x_train[:train_rows], x_val[:val_rows]
        if c.subsample_train is not None and c.subsample_train < len(x_train):
            # head of the train split after a seeded shuffle
            g = torch.Generator().manual_seed(128)
            order = torch.randperm(len(x_train), generator=g)
            x_train = x_train[order[: c.subsample_train]]

        self.info = info
        self.d_in = x_train.shape[1]
        self.x_train = x_train.to(self.device)
        self.x_val = x_val.to(self.device)

    def _smoke_rows(self, key):
        train_cfg, _ = load_configs()
        return train_cfg["smoke"][key]

    def _build(self):
        _, model_cfg = load_configs()
        self.model = build_model(
            d_in=self.d_in,
            cfg=model_cfg,
            variant=self.cfg.arch_variant,
            trunk=self.cfg.trunk,
            bottleneck=self.cfg.bottleneck,
            smoke=self.cfg.smoke,
            time_input=self.cfg.method != "dae",
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    # -- one epoch ---------------------------------------------------------

    def _forward(self, z, t):
        if self.cfg.precision == "fp32":
            return self.model(z, t)
        if self.cfg.precision != "bf16":
            raise ValueError(f"unknown precision {self.cfg.precision!r}")
        # no GradScaler needed, bf16 has the same exponent range as fp32.
        # the output is cast back so the loss is computed in fp32
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16):
            return self.model(z, t).float()

    def _train_epoch(self, epoch):
        """one epoch. returns the per-batch stats as an (n_batches, 3) tensor
        on the device, or None when instrumentation is off.

        the stats stay on the device so the loop doesnt wait on a copy every
        batch, `run` moves them once per interval.
        """
        c = self.cfg
        n = len(self.x_train)
        g = _epoch_generator(c.seed, epoch, self.device)

        order = torch.randperm(n, generator=g, device=self.device)
        n_batches = n // c.batch_size  # leftover rows are dropped
        stats = []

        self.model.train()
        for b in range(n_batches):
            idx = order[b * c.batch_size : (b + 1) * c.batch_size]
            x = self.x_train[idx]
            eps = torch.randn(x.shape, generator=g, device=self.device)
            if c.method == "dae":
                out = self._forward(x + c.dae_sigma * eps, None)
                per_sample = ((out - x) ** 2).mean(dim=1)
            else:
                t = interpolant.sample_t(len(x), c.delta, generator=g, device=self.device)
                z = interpolant.interpolate(x, eps, t)
                out = self._forward(z, t)
                per_sample = conversions.loss_per_sample(
                    c.pred_mode, c.loss_space, out, z, t, x, eps
                )
            loss = per_sample.mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
                if c.instrument
                else None
            )
            self.optimizer.step()

            if c.instrument:
                ps = per_sample.detach()
                stats.append(torch.stack((ps.median(), ps.max(), grad_norm)))
        return torch.stack(stats) if stats else None

    # -- validation --------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch):
        """loss over the full val split at each fixed t, in the cell's loss space."""
        c = self.cfg
        self.model.eval()
        if c.method == "dae":
            # one fixed noise draw for every check, so the stopping rule
            # compares models and not noise draws
            g = _epoch_generator(c.seed + c.eps_seed_base, 0, self.device, tag=1)
            eps = torch.randn(self.x_val.shape, generator=g, device=self.device)
            out = self.model(self.x_val + c.dae_sigma * eps, None)
            value = ((out - self.x_val) ** 2).mean()
            return [{"epoch": epoch, "t": "", "loss": float(value)}]
        # fresh eps per evaluation, seeded by (seed, epoch)
        g = _epoch_generator(c.seed + c.eps_seed_base, epoch, self.device, tag=1)
        eps = torch.randn(self.x_val.shape, generator=g, device=self.device)

        rows = []
        for t_value in c.val_t_grid:
            t = torch.full((len(self.x_val),), t_value, device=self.device)
            z = interpolant.interpolate(self.x_val, eps, t)
            out = self.model(z, t)
            value = conversions.loss(
                c.pred_mode, c.loss_space, out, z, t, self.x_val, eps
            )
            rows.append({"epoch": epoch, "t": t_value, "loss": float(value)})
        return rows

    # -- checkpointing -----------------------------------------------------

    def _checkpoint_payload(self):
        return {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "early_stop": {"best_val": self.best_val, "best_epoch": self.best_epoch,
                           "bad_checks": self.bad_checks},
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
        }

    def save_checkpoint(self):
        """write to .tmp then os.replace, so a crash cant leave a half file."""
        path = self.out_dir / "ckpt_last.pt"
        tmp = self.out_dir / "ckpt_last.pt.tmp"
        torch.save(self._checkpoint_payload(), tmp)
        os.replace(tmp, path)
        return path

    def _early_stop_update(self, val_loss):
        """record one validation check and keep the best model in ckpt_best.pt."""
        if val_loss < self.best_val:
            self.best_val, self.best_epoch, self.bad_checks = val_loss, self.epoch, 0
            tmp = self.out_dir / "ckpt_best.pt.tmp"
            torch.save({"epoch": self.epoch, "val_loss": val_loss,
                        "model": self.model.state_dict()}, tmp)
            os.replace(tmp, self.out_dir / "ckpt_best.pt")
        else:
            self.bad_checks += 1

    def stopped_early(self):
        c = self.cfg
        return c.method == "dae" and self.bad_checks >= c.patience

    def load_checkpoint(self):
        path = self.out_dir / "ckpt_last.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"]
        stop = ckpt.get("early_stop")
        if stop:
            self.best_val = stop["best_val"]
            self.best_epoch = stop["best_epoch"]
            self.bad_checks = stop["bad_checks"]
        rng = ckpt["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        return True

    # -- logging -----------------------------------------------------------

    def _append_csv(self, name, rows, header):
        path = self.out_dir / name
        exists = path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if not exists:
                w.writeheader()
            w.writerows(rows)

    def _instrument_row(self, epoch, stats):
        """aggregate an interval's per-batch stats into one row.

        diagnostic only, to see whether the big loss scale of the mismatched
        pred/loss cells reaches the parameter updates.
        """
        med = np.array([s[0] for s in stats])
        mx = np.array([s[1] for s in stats])
        gn = np.array([s[2] for s in stats])
        return {
            "epoch": epoch,
            "batches": len(stats),
            "loss_median_med": float(np.median(med)),
            "loss_max_med": float(np.median(mx)),
            "loss_max_p99": float(np.percentile(mx, 99)),
            "grad_norm_med": float(np.median(gn)),
            "grad_norm_max": float(gn.max()),
        }

    # -- the loop ----------------------------------------------------------

    def run(self):
        c = self.cfg
        resumed = self.load_checkpoint()
        start = self.epoch
        if resumed:
            self._log(f"resumed at epoch {start}")

        if self.stopped_early():
            return self.epoch

        pending = []
        for epoch in range(start, c.epochs):
            stats = self._train_epoch(epoch)
            if stats is not None:
                pending.append(stats)
            self.epoch = epoch + 1

            if self.epoch % c.eval_every == 0 or self.epoch == c.epochs:
                val_rows = self._validate(self.epoch)
                self._append_csv("val_loss.csv", val_rows, ["epoch", "t", "loss"])
                if c.method == "dae":
                    self._early_stop_update(val_rows[0]["loss"])
                if c.instrument and pending:
                    # the only device to host copy of the stats in the interval
                    rows = torch.cat(pending).cpu().tolist()
                    row = self._instrument_row(self.epoch, rows)
                    self._append_csv("train_stats.csv", [row], list(row))
                    pending = []
                self.save_checkpoint()
                self.heartbeat()
                self._log(f"epoch {self.epoch}/{c.epochs}")
                if self.stopped_early():
                    self._log(f"early stop at epoch {self.epoch}; best epoch "
                              f"{self.best_epoch}, val loss {self.best_val:.6g}")
                    break

        self.save_checkpoint()
        return self.epoch

    def _log(self, message):
        with open(self.out_dir / "train.log", "a") as f:
            f.write(message + "\n")


def train_cell(cfg: TrainConfig, out_dir, device=None, heartbeat=None):
    trainer = Trainer(cfg, out_dir, device=device, heartbeat=heartbeat)
    return trainer.run()
