"""probe scores and their standard errors.

analytic standard errors, from n_test and not from repeated runs:

    classification   SE(acc)  = sqrt(p*(1-p)/n)
    regression       SE(MSE)  = sd(e^2)/sqrt(n)

two extras:

1. per-row errors are returned as well as the aggregate, so two methods can
   be compared paired on the same rows. row difficulty cancels in the
   difference, which makes the se on the difference a lot tighter than
   treating the two methods as independent.

2. mae is reported next to mse. the regression targets are heavy-tailed, so
   mse and its se are driven by a few outliers. mae is more stable on the
   same residuals. mse is still the main metric.
"""

import numpy as np


def _as_1d(a):
    return np.asarray(a).reshape(-1)


# --------------------------------------------------------------------------
# per-row errors, retained so comparisons can be paired
# --------------------------------------------------------------------------


def squared_errors(y_true, y_pred):
    return (_as_1d(y_true) - _as_1d(y_pred)) ** 2


def correctness(y_true, y_pred):
    return (_as_1d(y_true) == _as_1d(y_pred)).astype(np.float64)


# --------------------------------------------------------------------------
# single-method scores
# --------------------------------------------------------------------------


def classification_score(y_true, y_pred):
    c = correctness(y_true, y_pred)
    n = len(c)
    p = float(c.mean())
    return {
        "metric": "accuracy",
        "value": p,
        "se": float(np.sqrt(p * (1.0 - p) / n)) if n else float("nan"),
        "n": n,
    }


def regression_score(y_true, y_pred):
    """mse with its se, plus rmse and mae with theirs.

    SE(RMSE) comes from SE(MSE) by the delta method, since RMSE = sqrt(MSE):
    SE(RMSE) = SE(MSE) / (2*RMSE).
    """
    e = _as_1d(y_true) - _as_1d(y_pred)
    e2 = e**2
    abs_e = np.abs(e)
    n = len(e)

    mse = float(e2.mean())
    se_mse = float(e2.std(ddof=1) / np.sqrt(n))
    rmse = float(np.sqrt(mse))
    se_rmse = float(se_mse / (2.0 * rmse)) if rmse > 0 else float("nan")

    return {
        "metric": "mse",
        "value": mse,
        "se": se_mse,
        "rmse": rmse,
        "se_rmse": se_rmse,
        "mae": float(abs_e.mean()),
        "se_mae": float(abs_e.std(ddof=1) / np.sqrt(n)),
        "n": n,
    }


def score(task_type, y_true, y_pred):
    """dispatch on task type."""
    if task_type == "regression":
        return regression_score(y_true, y_pred)
    return classification_score(y_true, y_pred)


def per_row_errors(task_type, y_true, y_pred):
    """the per-row quantity to store for a later paired comparison."""
    if task_type == "regression":
        return squared_errors(y_true, y_pred)
    return correctness(y_true, y_pred)


# --------------------------------------------------------------------------
# paired comparison between two methods scored on the same rows
# --------------------------------------------------------------------------


def paired_difference(errors_a, errors_b, task_type="regression"):
    """difference between two methods scored on the same rows.

    pass the per-row values from per_row_errors: squared errors for regression,
    correct/incorrect for classification. the sign is "a minus b", so for
    regression a negative delta means a has the lower error and for
    classification a positive delta means a is more accurate.
    """
    a, b = _as_1d(errors_a), _as_1d(errors_b)
    if len(a) != len(b):
        raise ValueError(
            f"paired comparison needs identical rows: got {len(a)} and {len(b)}"
        )
    d = a - b
    n = len(d)
    delta = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    out = {
        "delta": delta,
        "se": se,
        "n": n,
        "z": float(delta / se) if se > 0 else float("nan"),
        "metric": "d_mse" if task_type == "regression" else "d_accuracy",
    }
    return out


def unpaired_se(se_a, se_b):
    """the error bar if the pairing is ignored."""
    return float(np.sqrt(se_a**2 + se_b**2))
