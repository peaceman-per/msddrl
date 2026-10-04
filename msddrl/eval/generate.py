"""the generate stage of exp 1.

sample n_train rows with the ode sampler, invert the training transforms and
write samples.csv. there is no probe, the eval stage is done when samples.csv
exists.

exp 1 is the only place the target is in the model input, so it gets
generated and decoded with everything else.
"""

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from msddrl.data.cat_converter import CatConverter
from msddrl.data.loader import _processed_root, load_info
from msddrl.model.interpolant import load_config as load_model_config
from msddrl.model.sampler import sample


def _columns(info):
    cols = list(info["column_names"])
    num = [cols[i] for i in info["num_col_idx"]]
    cat = [cols[i] for i in info["cat_col_idx"]]
    target = info["target_col"]
    return cols, num, cat, target


def decode(encoded, info, transforms, include_target=True):
    """encoding space -> a dataframe in the original columns and units."""
    cols, num_cols, cat_cols, target = _columns(info)
    regression = info["task_type"] == "regression"

    n_num = len(num_cols) + (1 if (include_target and regression) else 0)
    num_block = encoded[:, :n_num].cpu().numpy()
    cat_block = encoded[:, n_num:]

    cardinalities = list(info["cat_cardinalities"])
    if include_target and not regression:
        cardinalities = cardinalities + [info["target_cardinality"]]

    frame = {}

    # numerics: invert the quantile transformer fitted on train
    if num_cols or n_num:
        numeric = num_block[:, : len(num_cols)]
        if len(num_cols):
            numeric = transforms["numeric"].inverse_transform(numeric)
            for j, name in enumerate(num_cols):
                frame[name] = numeric[:, j]
        if include_target and regression:
            scaled = num_block[:, len(num_cols):len(num_cols) + 1]
            frame[target] = transforms["target"].inverse_transform(scaled).ravel()

    # categoricals: CatConverter's inverse, then the fitted vocabulary
    if cardinalities:
        codes, casting_rate = CatConverter(cardinalities).decode(cat_block)
        codes = codes.cpu().numpy()
        if cat_cols:
            categories = transforms["categorical"].categories_
            for j, name in enumerate(cat_cols):
                levels = categories[j]
                frame[name] = levels[np.clip(codes[:, j], 0, len(levels) - 1)]
        if include_target and not regression:
            levels = transforms["target"].categories_[0]
            frame[target] = levels[np.clip(codes[:, -1], 0, len(levels) - 1)]
    else:
        casting_rate = 0.0

    return pd.DataFrame(frame)[[c for c in cols if c in frame]], casting_rate


def run_generate_stage(cell, out_dir, device=None, n_samples=None):
    """sample, decode and write samples.csv for one exp 1 cell."""
    info = load_info(cell.dataset)
    model_cfg = load_model_config()
    processed = _processed_root() / cell.dataset
    transforms = joblib.load(processed / "transforms.joblib")

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(out_dir / "ckpt_last.pt", map_location=device, weights_only=False)

    from msddrl.model.mlp import build as build_model

    d_in = ckpt["model"]["proj.weight"].shape[1]
    model = build_model(
        d_in=d_in, cfg=model_cfg, variant=cell.arch, trunk=cell.trunk,
        bottleneck=cell.bottleneck, smoke=cell.smoke,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    sampler_cfg = model_cfg["sampler"]
    steps = model_cfg["smoke"]["sampler_steps"] if cell.smoke else sampler_cfg["steps"]
    if n_samples is None:
        n_samples = (
            model_cfg["smoke"]["sampler_rows"] if cell.smoke
            else info["n_split"]["train"]
        )

    generated = sample(
        model, n_samples=n_samples, d_in=d_in, pred_mode=cell.pred,
        steps=steps, delta=model_cfg["interpolant"]["delta"],
        batch_size=sampler_cfg["batch_size"], device=device,
        generator=torch.Generator(device=device).manual_seed(cell.seed),
    )

    frame, casting_rate = decode(
        generated, info, transforms, include_target=cell.target_in_input
    )
    path = out_dir / "samples.csv"
    frame.to_csv(path, index=False)

    (out_dir / "generate.json").write_text(
        yaml.safe_dump(
            {
                "n_samples": int(len(frame)),
                "steps": int(steps),
                "pred_mode": cell.pred,
                "casting_rate": float(casting_rate),
            }
        )
    )
    return {"rows": len(frame), "casting_rate": casting_rate, "path": str(path)}
