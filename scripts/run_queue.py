#!/usr/bin/env python
"""run the experiment cells: train workers on the gpus, probe workers on cpu.

    python scripts/run_queue.py --experiments 1 2 3 4
    python scripts/run_queue.py --experiments 3 --seeds 2

workers claim a cell stage with a lock file (O_CREAT|O_EXCL), so two workers
never run the same thing. a lock with a heartbeat older than 30 min counts as
dead and can be taken over. results go in the cell's own folder.
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from msddrl.train.cell import (  # noqa: E402
    MAX_RETRIES,
    Claim,
    eval_is_claimable,
    is_done,
    read_status,
    set_state,
)
from msddrl.train.queue import enumerate_cells  # noqa: E402
from msddrl.train.trainer import TrainConfig, Trainer, load_configs  # noqa: E402


def manifest_path(smoke):
    return REPO_ROOT / ("runs_smoke" if smoke else "runs") / "manifest.jsonl"


def write_config(cell):
    d = cell.root()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "config.json"
    if not path.exists():
        path.write_text(json.dumps(cell.resolved(), indent=2))
    return d


def config_matches(cell):
    """a changed config must not reuse an old result."""
    path = cell.root() / "config.json"
    if not path.exists():
        return False
    stored = json.loads(path.read_text())
    return stored.get("cell_id") == cell.cell_id


def check_widths(model, cell):
    """check the built model has the widths the cell config says."""
    if cell.smoke:
        return
    built = [layer.out_features for layer in model.hidden]
    declared = list(cell.trunk or []) + [cell.bottleneck]
    if cell.trunk is None:
        built, declared = built[-1:], declared[-1:]
    if built != declared:
        raise RuntimeError(
            f"cell {cell.cell_id} declares widths {declared} but the model "
            f"was built with {built}"
        )


def cell_is_finished(cell):
    d = cell.root()
    if not d.exists() or not config_matches(cell):
        return False
    results = d / cell.results_name()
    return is_done(d) and results.exists()


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def train_config(cell):
    train_cfg, model_cfg = load_configs()
    return TrainConfig.from_configs(
        cell.dataset, train_cfg, model_cfg, smoke=cell.smoke,
        pred_mode=cell.pred, loss_space=cell.loss,
        arch_variant=cell.arch, trunk=cell.trunk,
        bottleneck=cell.bottleneck, seed=cell.seed,
        target_in_input=cell.target_in_input,
        subsample_train=cell.subsample_train,
        epochs=cell.epochs, eval_every=cell.eval_every,
        precision=cell.precision, fold=cell.fold,
        method=cell.method, dae_sigma=cell.dae_sigma, patience=cell.patience,
    )


REUSED_FILES = ("val_loss.csv", "train_stats.csv")


def find_reusable(cell, runs_root=None):
    """find a finished run in another experiment with the same training config.

    same model_id means same weights (e.g. exp 4 seed 1 vs exp 2), so the
    checkpoint can be copied instead of retrained. the stored config is also
    compared field by field in case of a hash collision.
    """
    if cell.smoke:
        return None
    runs_root = Path(runs_root) if runs_root else REPO_ROOT / "runs"
    wanted = cell.training_config()
    own = cell.root().resolve()
    for path in sorted(runs_root.glob("exp*/*/config.json")):
        src = path.parent
        if src.resolve() == own:
            continue
        try:
            stored = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if stored.get("model_id") != cell.model_id:
            continue
        if any(stored.get(k) != v for k, v in wanted.items()):
            continue
        if read_status(src)["train"]["state"] != "done":
            continue
        if not (src / "ckpt_last.pt").exists():
            continue
        return src
    return None


def reuse_checkpoint(cell, src, dest):
    """copy a finished run's checkpoint and training logs into this cell."""
    ckpt = torch.load(src / "ckpt_last.pt", map_location="cpu", weights_only=False)
    if ckpt["epoch"] != cell.epochs:
        raise RuntimeError(
            f"{src.name} stopped at epoch {ckpt['epoch']}, cell wants {cell.epochs}"
        )
    tmp = dest / "ckpt_last.pt.tmp"
    shutil.copy2(src / "ckpt_last.pt", tmp)
    os.replace(tmp, dest / "ckpt_last.pt")
    for name in REUSED_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    (dest / "reused_from.json").write_text(json.dumps(
        {"cell_id": src.name, "experiment_dir": src.parent.name,
         "model_id": cell.model_id}, indent=2))
    return ckpt["epoch"]


def run_train_stage(cell, device, worker, reuse=True):
    d = write_config(cell)
    manifest = manifest_path(cell.smoke)
    claim = Claim(d, "train", worker=worker)
    if not claim.acquire():
        return "skipped"

    try:
        src = None
        if reuse and not (d / "ckpt_last.pt").exists():
            src = find_reusable(cell)
        if src is not None:
            epoch = reuse_checkpoint(cell, src, d)
            set_state(d, "train", "done", manifest=manifest,
                      cell_id=cell.cell_id, worker=worker, epoch=epoch,
                      reused_from=f"{src.parent.name}/{src.name}")
            return f"reused {src.parent.name}/{src.name}"

        set_state(d, "train", "running", manifest=manifest,
                  cell_id=cell.cell_id, worker=worker)
        cfg = train_config(cell)
        trainer = Trainer(cfg, d, device=device, heartbeat=claim.heartbeat)
        check_widths(trainer.model, cell)
        epoch = trainer.run()
        set_state(d, "train", "done", manifest=manifest,
                  cell_id=cell.cell_id, worker=worker, epoch=epoch)
        return "done"
    except Exception:
        status = read_status(d)
        retries = status["train"].get("retries", 0) + 1
        state = "failed" if retries < MAX_RETRIES else "dead"
        set_state(d, "train", state, manifest=manifest, cell_id=cell.cell_id,
                  worker=worker, retries=retries)
        (d / "train.log").open("a").write(traceback.format_exc())
        return state
    finally:
        claim.release()


def run_eval_stage(cell, worker, device="cpu"):
    d = cell.root()
    manifest = manifest_path(cell.smoke)
    if not eval_is_claimable(d):
        return "waiting"

    claim = Claim(d, "eval", worker=worker)
    if not claim.acquire():
        return "skipped"

    try:
        set_state(d, "eval", "running", manifest=manifest,
                  cell_id=cell.cell_id, worker=worker)
        if cell.eval_spec.kind == "generate":
            from msddrl.eval.generate import run_generate_stage

            run_generate_stage(cell, d, device=device)
        else:
            from msddrl.eval.probe import (
                PROBE_MAX_ITER,
                SMOKE_MAX_ITER,
                run_probe_stage,
            )
            from msddrl.model.mlp import build as build_model

            _, model_cfg = load_configs()
            # a dae is scored at its best validation check, not the last one
            ckpt_name = "ckpt_best.pt" if cell.method == "dae" else "ckpt_last.pt"
            ckpt = torch.load(d / ckpt_name, map_location=device,
                              weights_only=False)
            model = build_model(
                d_in=ckpt["model"]["proj.weight"].shape[1], cfg=model_cfg,
                variant=cell.arch, trunk=cell.trunk,
                bottleneck=cell.bottleneck, smoke=cell.smoke,
                time_input=cell.method != "dae",
            ).to(device)
            model.load_state_dict(ckpt["model"])

            run_probe_stage(
                model, cell.dataset, cell.eval_spec.pairs(),
                cell.eval_spec.score_split, d,
                include_target=cell.target_in_input,
                fold=cell.fold, device=device,
                max_iter=SMOKE_MAX_ITER if cell.smoke else PROBE_MAX_ITER,
                store_row_errors=cell.eval_spec.kind == "probe_point",
                heartbeat=claim.heartbeat,
            )
        set_state(d, "eval", "done", manifest=manifest,
                  cell_id=cell.cell_id, worker=worker)
        return "done"
    except Exception:
        status = read_status(d)
        retries = status["eval"].get("retries", 0) + 1
        state = "failed" if retries < MAX_RETRIES else "dead"
        set_state(d, "eval", state, manifest=manifest, cell_id=cell.cell_id,
                  worker=worker, retries=retries)
        (d / "train.log").open("a").write(traceback.format_exc())
        return state
    finally:
        claim.release()


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------


def train_worker(cells, device, worker, reuse=True):
    for cell in cells:
        if cell_is_finished(cell):
            continue
        status = read_status(cell.root())
        if status["train"]["state"] in ("done", "dead"):
            continue
        outcome = run_train_stage(cell, device, worker, reuse=reuse)
        print(f"[{worker}] train {cell.cell_id} {cell.dataset} -> {outcome}", flush=True)


def probe_worker(cells, worker, threads, training_done, poll=5.0, max_idle=60.0):
    os.environ.update(
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
    )
    idle = 0.0
    pending = list(cells)
    while pending:
        progressed = False
        for cell in list(pending):
            if cell_is_finished(cell):
                pending.remove(cell)
                continue
            status = read_status(cell.root())
            if status["eval"]["state"] in ("done", "dead"):
                pending.remove(cell)
                continue
            if status["train"]["state"] == "dead":
                pending.remove(cell)
                continue
            outcome = run_eval_stage(cell, worker)
            if outcome not in ("waiting", "skipped"):
                progressed = True
                print(f"[{worker}] eval {cell.cell_id} {cell.dataset} -> {outcome}",
                      flush=True)
        if progressed:
            idle = 0.0
        else:
            time.sleep(poll)
            # a cell still training will become claimable later, so only
            # start the idle timeout once all train workers have exited
            if not training_done.is_set():
                continue
            idle += poll
            if idle >= max_idle:
                break


def visible_gpus():
    if not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments", type=int, nargs="+", required=True)
    ap.add_argument("--probe-workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--devices", nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--order", choices=("cheap", "expensive"), default="cheap",
                    help="cheap (default): most cells done if interrupted. "
                         "expensive: longest first, for uneven GPUs.")
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                    help="training precision for every cell. bf16 cells hash "
                         "differently, so they get their own directories.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="only the cells with these seeds")
    ap.add_argument("--no-reuse", action="store_true",
                    help="train every cell, even when a finished run with the "
                         "same model_id exists in another experiment")
    args = ap.parse_args(argv)

    overrides = {"precision": args.precision} if args.precision != "fp32" else None
    cells = enumerate_cells(args.experiments, smoke=args.smoke,
                            overrides=overrides, order=args.order)
    if args.seeds:
        cells = [c for c in cells if c.seed in args.seeds]
    remaining = [c for c in cells if not cell_is_finished(c)]
    print(f"{len(cells)} cells, {len(remaining)} outstanding")

    if args.dry_run:
        reusable = {c.cell_id: find_reusable(c) for c in remaining
                    if not args.no_reuse and not (c.root() / "ckpt_last.pt").exists()}
        n_reuse = sum(1 for v in reusable.values() if v is not None)
        print(f"{n_reuse} of them can reuse a finished checkpoint")
        for c in remaining[:20]:
            src = reusable.get(c.cell_id)
            print(f"  exp{c.experiment} {c.cell_id} {c.dataset:12s} "
                  f"{c.arch or '-':7s} pred={c.pred} loss={c.loss} bn={c.bottleneck} "
                  f"seed={c.seed} fold={c.fold} {c.extra.get('variant', '')} "
                  f"{c.precision}" + (f" <- {src.parent.name}/{src.name}" if src else ""))
        if len(remaining) > 20:
            print(f"  ... and {len(remaining)-20} more")
        return 0

    if not remaining:
        print("nothing to do")
        return 0

    devices = args.devices or visible_gpus()
    threads = max(1, (os.cpu_count() or 4) // max(1, args.probe_workers))

    # cuda cant be re-initialised in a forked child, so spawn
    ctx = get_context("spawn")
    training_done = ctx.Event()
    trainers, probers = [], []
    for i, device in enumerate(devices):
        # every worker gets all cells, the claims decide who runs what
        trainers.append(ctx.Process(target=train_worker,
                                    args=(remaining, device, f"train{i}",
                                          not args.no_reuse)))
    for i in range(args.probe_workers):
        probers.append(ctx.Process(target=probe_worker,
                                   args=(remaining, f"probe{i}", threads,
                                         training_done)))

    for p in trainers + probers:
        p.start()
    for p in trainers:
        p.join()
    training_done.set()
    for p in probers:
        p.join()

    done = sum(1 for c in cells if cell_is_finished(c))
    print(f"{done}/{len(cells)} cells complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
