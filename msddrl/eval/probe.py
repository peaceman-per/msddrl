"""the probe stage.

fit a linear probe on the train-split embeddings and score it on the split
the cell's eval spec names. the probe is the same for every cell and method:

    classification  LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                       max_iter=1000)
    regression      Ridge(alpha=1.0)

both on standardised embeddings, scaler fitted on the train embeddings only.
regression is scored in the dataset's own units.

results.csv gets one row per (t_true, t_label) as it finishes and a restart
skips the pairs already there, so an interrupted 144-point sweep carries on
where it stopped. embeddings are not written to disk.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from msddrl.data.loader import load_fold, load_split
from msddrl.eval import metrics
from msddrl.eval.extract import extract, noise_matrix

RESULT_COLUMNS = [
    "t_true",
    "t_label",
    "score_split",
    "metric",
    "value",
    "se",
    "rmse",
    "se_rmse",
    "mae",
    "se_mae",
    "n",
]

PROBE_MAX_ITER = 1000
SMOKE_MAX_ITER = 50


def build_probe(task_type, max_iter=PROBE_MAX_ITER):
    if task_type == "regression":
        return Ridge(alpha=1.0)
    return LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=max_iter
    )


def probe_embeddings(task_type, z_train, y_train, z_score, y_score,
                     max_iter=PROBE_MAX_ITER):
    """fit the probe on train embeddings and score the scored split.

    returns (score, prediction).
    """
    z_train = np.asarray(z_train)
    z_score = np.asarray(z_score)
    scaler = StandardScaler().fit(z_train)
    probe = build_probe(task_type, max_iter=max_iter)
    probe.fit(scaler.transform(z_train), y_train)
    prediction = probe.predict(scaler.transform(z_score))
    return metrics.score(task_type, y_score, prediction), prediction


def completed_pairs(results_path):
    """pairs already scored, so a resumed sweep skips them."""
    path = Path(results_path)
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            done.add((round(float(row["t_true"]), 6), round(float(row["t_label"]), 6)))
    return done


def _append(results_path, row):
    path = Path(results_path)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


def _load(dataset, split, include_target, fold=None):
    if fold is None:
        return load_split(dataset, split, include_target)
    return load_fold(dataset, fold, split, include_target)


def run_probe_stage(
    model,
    dataset,
    pairs,
    score_split,
    out_dir,
    task_type=None,
    include_target=False,
    fold=None,
    device="cpu",
    max_iter=PROBE_MAX_ITER,
    store_row_errors=False,
    heartbeat=None,
):
    """extract and probe every pair, appending one results row per pair.

    store_row_errors saves the per-row errors next to the results so two
    methods can be compared paired later. only used for single-point evals.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"
    heartbeat = heartbeat or (lambda: None)

    train_split = "train"
    x_train, y_train, idx_train, info = _load(dataset, train_split, include_target, fold)
    x_score, y_score, idx_score, _ = _load(dataset, score_split, include_target, fold)
    task_type = task_type or info["task_type"]

    e_train = noise_matrix(train_split, len(x_train), x_train.shape[1])
    e_score = noise_matrix(score_split, len(x_score), x_score.shape[1])

    already = completed_pairs(results_path)
    written = 0

    for t_true, t_label in pairs:
        key = (round(float(t_true), 6), round(float(t_label), 6))
        if key in already:
            continue

        z_train, idx_a = extract(
            model, x_train, idx_train, t_true, t_label, e_train, device=device
        )
        z_score, idx_b = extract(
            model, x_score, idx_score, t_true, t_label, e_score, device=device
        )

        # the rows being scored must be the rows that were labelled
        if not np.array_equal(idx_b, idx_score):
            raise AssertionError(
                f"{dataset}/{score_split}: embedding index diverged from labels"
            )
        score, prediction = probe_embeddings(
            task_type, z_train.numpy(), y_train, z_score.numpy(), y_score,
            max_iter=max_iter,
        )
        row = {
            "t_true": t_true,
            "t_label": t_label,
            "score_split": score_split,
            **score,
        }
        _append(results_path, row)
        written += 1

        if store_row_errors:
            np.save(
                out_dir / f"row_errors_{key[0]:.6f}_{key[1]:.6f}.npy",
                metrics.per_row_errors(task_type, y_score, prediction),
            )
        heartbeat()

    return {"written": written, "total": len(pairs), "results": str(results_path)}


def stage_is_complete(results_path, pairs):
    """done when every pair in the eval spec has a row."""
    done = completed_pairs(results_path)
    return all(
        (round(float(a), 6), round(float(b), 6)) in done for a, b in pairs
    )


def probe_sanity_check(n=512, d=16, seed=128):
    """sanity check: a linearly separable embedding must probe above 0.99.

    catches label misalignment and inverted targets.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    z = rng.normal(size=(n, d)) + 4.0 * y[:, None]
    scaler = StandardScaler().fit(z)
    probe = build_probe("binclass")
    probe.fit(scaler.transform(z), y)
    accuracy = float((probe.predict(scaler.transform(z)) == y).mean())
    return accuracy
