# msddrl

code for the four experiments. each experiment is a config in
`configs/experiments/`, and `scripts/run_queue.py` trains and evaluates every
cell in it.

## setup

    pip install -r requirements.txt

the prepared datasets go in `data/processed/<dataset>/` (not in the repo).

## running

    python scripts/run_queue.py --experiments 1 2 3 4

run them in that order, exp 3 and exp 4 reuse checkpoints from earlier
experiments where the training config is the same. `--dry-run` lists the cells
without running anything, `--seeds` picks seeds, `--devices` picks gpus.

outputs go in `runs/exp<k>/<cell_id>/`: the config, checkpoint, validation
loss and `results.csv` (or `samples.csv` for exp 1). a run that is stopped can
be started again with the same command and carries on where it was.

## layout

    configs/               training, model and data constants + the experiments
    msddrl/model/          the denoiser, interpolant, x/eps/v conversions, sampler
    msddrl/train/          cells, the queue and the training loop
    msddrl/eval/           extraction, linear probe, metrics, exp 1 generation
    msddrl/data/           loading the prepared splits
    scripts/run_queue.py   entry point

the model and the categorical encoding are based on
[TabRep](https://github.com/jacobyhsi/TabRep) (MIT licence, see LICENSE).
