"""turn the experiment configs into cells and order them.

configs/experiments/exp<k>.yaml gives `axes`, and the cells are their
cartesian product minus whatever `constraints` removes. a config can also
list its cells one by one under `cells:` (exp 4, where the tuned setting is
picked per dataset), or give several products under `blocks:` (exp 3, where
the widths differ by dataset). blocks can overlap, duplicates are kept once.

experiments run in order 1, 2, 3, 4 and within one the cheapest cells go
first, so if the run is interrupted as many cells as possible are finished.
"""

import dataclasses
import itertools
from pathlib import Path

import yaml

from msddrl.data.loader import load_folds_meta, load_info
from msddrl.model.interpolant import grid_pairs
from msddrl.train.cell import Cell, EvalSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiments"

AXIS_FIELDS = ("dataset", "pred", "loss", "bottleneck", "arch", "seed")


def load_experiment(experiment, directory=EXPERIMENT_DIR):
    path = Path(directory) / f"exp{experiment}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no config for experiment {experiment}: {path} missing"
            + (
                " (exp4.yaml is written from the exp 2 results)"
                if experiment == 4
                else ""
            )
        )
    with open(path) as f:
        return yaml.safe_load(f)


def _architecture_variant(name, smoke=False):
    with open(REPO_ROOT / "configs" / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    if smoke:
        return {
            "trunk": list(model_cfg["smoke"]["trunk"]),
            "bottleneck": model_cfg["smoke"]["bottlenecks"][0],
        }
    variant = model_cfg["architecture"]["variants"][name]
    return {"trunk": list(variant["trunk"]), "bottleneck": variant["bottleneck"]}


def _eval_spec(spec, smoke=False):
    spec = dict(spec or {})
    kind = spec.get("kind", "probe_point")
    grid = spec.get("grid")

    if kind == "probe_grid":
        pairs = grid_pairs()
        if grid == "interpolant" or grid is None:
            pass
        elif isinstance(grid, list):
            pairs = [tuple(p) for p in grid]
        if smoke:
            # smoke run uses a 2x2 grid of the two extreme values
            values = sorted({p[0] for p in pairs})
            extremes = [values[-1], values[0]]
            pairs = [(a, b) for a in extremes for b in extremes]
        return EvalSpec(
            kind=kind, grid=pairs, score_split=spec.get("score_split", "val")
        )

    return EvalSpec(
        kind=kind,
        point=tuple(spec["point"]) if spec.get("point") else None,
        score_split=spec.get("score_split", "val"),
        fold=spec.get("fold"),
    )


def _skipped(values, rules):
    """true when `values` matches any skip rule.

    a rule maps an axis name to a value or list of values, and matches when
    every key in it matches. used to cut corners out of the product, e.g.
    exp 3's two narrowest bottlenecks only make sense on adult.
    """
    for rule in rules or []:
        wanted = dict(rule)
        if not wanted:
            continue
        if all(
            values.get(axis) in (v if isinstance(v, list) else [v])
            for axis, v in wanted.items()
        ):
            return True
    return False


CELL_FIELDS = {f.name for f in dataclasses.fields(Cell)} - {"eval_spec", "extra"}


def _listed_cells(config, defaults, eval_default, smoke=False, overrides=None):
    """cells given one by one under `cells:`.

    each entry names Cell fields directly, plus t_true/t_label for its probe
    point. any other key (exp 4's `variant`, `selected_from`) goes in `extra`,
    which is hashed into cell_id but not model_id, so the tuned and naive
    cells get their own folders but can share a checkpoint.
    """
    cells = []
    for entry in config["cells"]:
        entry = dict(entry)
        spec = dict(eval_default)
        spec.update(entry.pop("eval", {}) or {})
        if "t_true" in entry or "t_label" in entry:
            spec["point"] = [entry.pop("t_true"), entry.pop("t_label")]
        kwargs = dict(defaults)
        extra = dict(kwargs.pop("extra", {}) or {})
        for key, value in entry.items():
            if key in CELL_FIELDS:
                kwargs[key] = value
            elif value is not None:
                extra[key] = value
        kwargs.update(overrides or {})
        kwargs.setdefault("loss", kwargs.get("pred"))
        kwargs.setdefault("bottleneck", 512)
        kwargs["experiment"] = config["experiment"]
        kwargs["smoke"] = smoke
        kwargs["extra"] = extra
        kwargs["eval_spec"] = _eval_spec(spec, smoke=smoke)
        cells.append(Cell(**kwargs))

    ids = [c.cell_id for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError(f"exp{config['experiment']}: duplicate cells in `cells:`")
    return cells


def expand(experiment, directory=EXPERIMENT_DIR, smoke=False, overrides=None):
    """every cell declared by one experiment config."""
    config = load_experiment(experiment, directory)
    defaults = dict(config.get("defaults") or {})
    eval_default = defaults.pop("eval", {})
    forms = [k for k in ("axes", "cells", "blocks") if config.get(k)]
    if len(forms) > 1:
        raise ValueError(f"exp{experiment}: give one of `axes`, `cells` or `blocks`, "
                         f"not {' and '.join(forms)}")
    if "cells" in config:
        return _listed_cells(config, defaults, eval_default, smoke, overrides)
    if "blocks" in config:
        cells, seen = [], set()
        for block in config["blocks"]:
            for cell in _product_cells(config, defaults, eval_default,
                                       block.get("axes"), block.get("constraints"),
                                       smoke, overrides):
                if cell.cell_id not in seen:
                    seen.add(cell.cell_id)
                    cells.append(cell)
        return cells
    return _product_cells(config, defaults, eval_default, config.get("axes"),
                          config.get("constraints"), smoke, overrides)


def _product_cells(config, defaults, eval_default, axes, constraints,
                   smoke=False, overrides=None):
    """cartesian product of `axes`, minus what `constraints` rules out."""
    axes = dict(axes or {})
    constraints = dict(constraints or {})
    skip_rules = constraints.get("skip") or []

    names = [a for a in AXIS_FIELDS if a in axes]
    combos = itertools.product(*(axes[n] for n in names)) if names else [()]

    cells = []
    for combo in combos:
        values = dict(zip(names, combo))
        if constraints.get("loss") == "same_as_pred":
            values["loss"] = values["pred"]
        if _skipped(values, skip_rules):
            continue

        spec = dict(eval_default)
        spec.update(values.pop("eval", {}) or {})

        kwargs = dict(defaults)
        kwargs.update(values)
        kwargs.update(overrides or {})
        kwargs.setdefault("loss", kwargs.get("pred"))
        if kwargs.get("arch"):
            # put the named variant's widths into the cell config, since
            # they are hashed and the folder name should match what trains
            variant = _architecture_variant(kwargs["arch"], smoke=smoke)
            kwargs["trunk"] = list(variant["trunk"])
            kwargs["bottleneck"] = variant["bottleneck"]
        kwargs.setdefault("bottleneck", 512)
        kwargs["experiment"] = config["experiment"]
        kwargs["smoke"] = smoke
        kwargs["eval_spec"] = _eval_spec(spec, smoke=smoke)
        cells.append(Cell(**kwargs))
    return cells


def _n_train(cell):
    info = load_info(cell.dataset)
    if cell.fold is None:
        return info["n_split"]["train"]
    folds = load_folds_meta(cell.dataset)
    return next(f["n_train"] for f in folds["folds"] if f["fold"] == cell.fold)


def cell_cost(cell):
    """epochs * floor(n_train / batch), i.e. the number of iterations."""
    try:
        n_train = _n_train(cell)
    except FileNotFoundError:
        n_train = 0
    if cell.subsample_train:
        n_train = min(n_train, cell.subsample_train)
    return cell.epochs * (n_train // cell.batch_size)


def schedule_weight(cell):
    """cell_cost scaled by trunk width, used for longest-first ordering.

    iteration count ignores trunk width, but the exp 1 legacy trunk takes
    about 1.8x longer per iteration than the reduced one. only the rank
    order matters here.
    """
    trunk = cell.trunk or []
    width = (sum(trunk) / (len(trunk) * 1024)) if trunk else 1.0
    return cell_cost(cell) * max(width, 1e-9)


def enumerate_cells(experiments, directory=EXPERIMENT_DIR, smoke=False,
                    overrides=None, order="cheap"):
    """all cells for the given experiments, in run order.

    order="cheap" is the default. order="expensive" is longest first, which
    is better when the gpus have different speeds: otherwise the slow gpu can
    end up claiming a long cell last and run hours after the fast one.
    """
    out = []
    for experiment in sorted(experiments):
        cells = expand(experiment, directory, smoke=smoke, overrides=overrides)
        if order == "expensive":
            cells.sort(key=lambda c: (-schedule_weight(c), c.dataset, c.cell_id))
        else:
            cells.sort(key=lambda c: (cell_cost(c), c.dataset, c.cell_id))
        out.extend(cells)
    return out
