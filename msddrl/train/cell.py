"""cell identity, folder layout and status.

a cell is one training run plus its evaluation. it has two hashes:

    model_id  hash of the training config only (no experiment, no eval)
    cell_id   hash of the full config

two are needed so that a checkpoint can be shared between experiments, e.g.
exp 4 reusing an exp 2 model. the same config always maps to the same folder.
"""

import hashlib
import json
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

STATES = ("queued", "running", "done", "failed", "dead")
STAGES = ("train", "eval")
MAX_RETRIES = 3
STALE_AFTER = 30 * 60  # seconds


def _short_sha(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class EvalSpec:
    """how a cell is evaluated."""

    kind: str  # probe_grid | probe_point | generate
    score_split: str = "val"  # val for exp 2-3, test for exp 4
    grid: list = None  # probe_grid: list of (t_true, t_label)
    point: tuple = None  # probe_point: one (t_true, t_label)
    # rotating fold, exp 4 regression only. same value as Cell.fold, kept
    # here too so the eval dict keeps its shape (it is hashed into cell_id)
    fold: int = None

    def to_dict(self):
        d = asdict(self)
        if d.get("grid") is not None:
            d["grid"] = [list(p) for p in d["grid"]]
        if d.get("point") is not None:
            d["point"] = list(d["point"])
        return d

    def pairs(self):
        if self.kind == "probe_grid":
            return [tuple(p) for p in (self.grid or [])]
        if self.kind == "probe_point":
            return [tuple(self.point)] if self.point else []
        return []


@dataclass
class Cell:
    experiment: int
    dataset: str
    pred: str
    loss: str
    bottleneck: int
    seed: int
    eval_spec: EvalSpec
    arch: str = None  # named variant, exp 1 only
    trunk: list = None
    epochs: int = 15000
    batch_size: int = 4096
    eval_every: int = 150
    lr: float = 1e-4
    weight_decay: float = 5e-4
    delta: float = 1e-3
    t_sampling: str = "uniform"
    target_in_input: bool = False
    subsample_train: int = None
    smoke: bool = False
    precision: str = "fp32"  # fp32 | bf16 (autocast forward pass)
    # rotating fold (exp 4 regression only). the model trains on the fold's
    # train part and is scored on its test part, so it is part of the
    # training config
    fold: int = None
    # "ddae", or "dae" for the vanilla dae baseline in exp 4, with its noise
    # level and early stopping patience. only hashed for a dae
    method: str = "ddae"
    dae_sigma: float = None
    patience: int = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        spec_fold = self.eval_spec.fold
        if self.fold is None:
            self.fold = spec_fold
        elif spec_fold is None:
            self.eval_spec.fold = self.fold
        elif spec_fold != self.fold:
            raise ValueError(
                f"cell trains on fold {self.fold} but its eval spec names "
                f"fold {spec_fold}"
            )

    # -- identity ----------------------------------------------------------

    def training_config(self):
        """everything that decides the trained weights. no experiment, no eval.

        precision is only included when it isnt fp32 and fold only when set.
        they were added after runs had started and hashing the defaults
        would have changed every existing id.
        """
        cfg = {
            "dataset": self.dataset,
            "pred": self.pred,
            "loss": self.loss,
            "bottleneck": self.bottleneck,
            "arch": self.arch,
            "trunk": list(self.trunk) if self.trunk else None,
            "seed": self.seed,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "delta": self.delta,
            "t_sampling": self.t_sampling,
            "target_in_input": self.target_in_input,
            "subsample_train": self.subsample_train,
            "smoke": self.smoke,
        }
        if self.precision != "fp32":
            cfg["precision"] = self.precision
        if self.fold is not None:
            cfg["fold"] = self.fold
        if self.method != "ddae":
            cfg.update(method=self.method, dae_sigma=self.dae_sigma,
                       patience=self.patience)
        return cfg

    def full_config(self):
        cfg = dict(self.training_config())
        cfg["experiment"] = self.experiment
        cfg["eval"] = self.eval_spec.to_dict()
        cfg["eval_every"] = self.eval_every
        cfg["extra"] = self.extra
        return cfg

    @property
    def model_id(self):
        return _short_sha(self.training_config())

    @property
    def cell_id(self):
        return _short_sha(self.full_config())

    # -- layout ------------------------------------------------------------

    def root(self, base=None):
        base = Path(base) if base else REPO_ROOT / (
            "runs_smoke" if self.smoke else "runs"
        )
        return base / f"exp{self.experiment}" / self.cell_id

    def resolved(self):
        cfg = self.full_config()
        cfg["model_id"] = self.model_id
        cfg["cell_id"] = self.cell_id
        return cfg

    def results_name(self):
        return "samples.csv" if self.eval_spec.kind == "generate" else "results.csv"


# --------------------------------------------------------------------------
# status, written per cell; never a shared file
# --------------------------------------------------------------------------


def _now():
    return time.time()


def default_status():
    stamp = _now()
    return {
        stage: {"state": "queued", "epoch": 0, "rows_done": 0, "retries": 0,
                "updated": stamp}
        for stage in STAGES
    }


def read_status(cell_dir):
    path = Path(cell_dir) / "status.json"
    if not path.exists():
        return default_status()
    return json.loads(path.read_text())


def write_status(cell_dir, status):
    path = Path(cell_dir) / "status.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, path)


def set_state(cell_dir, stage, state, manifest=None, cell_id=None, worker=None, **fields):
    """move one stage to a new state and log the change in the manifest."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if state not in STATES:
        raise ValueError(f"unknown state {state!r}")

    cell_dir = Path(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    status = read_status(cell_dir)
    previous = status[stage]["state"]
    status[stage].update(state=state, updated=_now(), **fields)
    write_status(cell_dir, status)

    if manifest is not None:
        append_manifest(
            manifest,
            {
                "ts": _now(),
                "cell_id": cell_id or cell_dir.name,
                "stage": stage,
                "from": previous,
                "to": state,
                "worker": worker or f"{socket.gethostname()}:{os.getpid()}",
            },
        )
    return status


def is_done(cell_dir):
    """a cell is done only when both stages are done."""
    status = read_status(cell_dir)
    return all(status[stage]["state"] == "done" for stage in STAGES)


def append_manifest(manifest_path, record):
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_manifest(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# claims
# --------------------------------------------------------------------------


class ClaimError(RuntimeError):
    pass


class Claim:
    """exclusive claim on one stage of one cell.

    the lock file is created with O_CREAT|O_EXCL so only one worker can get
    it. the heartbeat is refreshed during training and probing, and a claim
    with a heartbeat older than STALE_AFTER can be taken over (thats how a
    crashed worker's cell gets picked up again).
    """

    def __init__(self, cell_dir, stage, worker=None, clock=time.time):
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        self.path = Path(cell_dir) / f"claim.{stage}.lock"
        self.stage = stage
        self.worker = worker or f"{socket.gethostname()}:{os.getpid()}"
        self.clock = clock
        self.held = False

    def _payload(self):
        return {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "worker": self.worker,
            "heartbeat": self.clock(),
        }

    def _write(self, fd):
        with os.fdopen(fd, "w") as f:
            json.dump(self._payload(), f)

    def acquire(self, steal_stale=True):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if steal_stale and self.is_stale():
                # previous holder probably crashed, take it over
                self.path.unlink(missing_ok=True)
                return self.acquire(steal_stale=False)
            return False
        self._write(fd)
        self.held = True
        return True

    def read(self):
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return None

    def is_stale(self):
        info = self.read()
        if info is None:
            return True
        return (self.clock() - info.get("heartbeat", 0)) > STALE_AFTER

    def heartbeat(self):
        if not self.held:
            return
        tmp = self.path.with_suffix(".lock.tmp")
        tmp.write_text(json.dumps(self._payload()))
        os.replace(tmp, self.path)

    def release(self):
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def __enter__(self):
        if not self.acquire():
            raise ClaimError(f"{self.path.name} already held")
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def eval_is_claimable(cell_dir):
    """eval can only start once training is done and a checkpoint exists."""
    cell_dir = Path(cell_dir)
    status = read_status(cell_dir)
    return (
        status["train"]["state"] == "done"
        and (cell_dir / "ckpt_last.pt").exists()
    )
